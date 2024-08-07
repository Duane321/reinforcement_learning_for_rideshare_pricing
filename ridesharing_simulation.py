import math
import itertools
from tqdm import tqdm
import json
import numpy as np
import utils
import torch

class WeeklySimulation:


    def __init__(self, current_week, learning_rate, initial_pricing_params, elasticity_params_dict, log_save_path):
        #self.current_day will be updated in update_gamma_distns
        self.current_day = 0

        self.current_week = current_week


        self.lr = learning_rate # lr for SGD
        self.pricing_params = initial_pricing_params

        # short-term rider's elasticity is around -0.05(i.e. one unit increase of request price
        # will cause 5% decrease of acceptance prob for riders)
        if not elasticity_params_dict:
            self.a_r = 1.5  # Rider acceptance probability parameter
            self.b_r = -0.2  # Rider acceptance probability parameter

            # short-term driver's elasticity is around -0.05 as well
            self.a_d = 1.5  # Driver acceptance probability parameter
            self.b_d = -0.3  # Driver acceptance probability parameter

            self.a_lambda = -0.001
            # avg. daily exposed price over 100 weeks is 22.5672
            self.b_lambda = 22.5672
        else:
            self.a_r = elasticity_params_dict['a_r']
            self.b_r = elasticity_params_dict['b_r']

            # short-term driver's elasticity is around -0.05 as well
            self.a_d = elasticity_params_dict['a_d']
            self.b_d = elasticity_params_dict['b_d']

            self.a_lambda = elasticity_params_dict['a_lambda']
            # avg. daily exposed price over 100 weeks is 22.5672
            self.b_lambda = elasticity_params_dict['b_lambda']

        self.log_save_path = log_save_path

        self.num_riders = 1000
        self.part_size = self.num_riders // 4
        self.daily_max_requests = 5
        
        self.num_drivers = 100
        self.num_subgrids_per_dim = 4
        self.num_subgrids = self.num_subgrids_per_dim ** 2
        self.subgrid_size = 1.0 / self.num_subgrids_per_dim

        # on avg. a driver can work for 8hrs
        self.mean_idle_time = 8 * 60

        self.match_interval_time = 30

        # sequential ID for total number of riders
        self.indices = torch.arange(self.num_riders)
        self.mask1 = self.indices < self.part_size
        self.mask2 = (self.indices >= self.part_size) & (self.indices < 2 * self.part_size)
        self.mask3 = (self.indices >= 2 * self.part_size) & (self.indices < 3 * self.part_size)
        #mask4 = indices >= 3 * self.part_size

        #number of different distributions for request time
        self.num_components = 3 
        #480, 1080, 1320, total 1440, 0 is placeholder for uniform distribution
        self.request_times_means = torch.tensor([8 * 60, 18 * 60, 22 * 60, 0])
        self.request_times_stds = torch.tensor([60, 60, 60, 0])

        #there are two types of commuters(morning and evening),each with 25% probability
        self.commuter_percentage, self.party_goer_percentage, self.sporadic_rider_percentage = 0.25, 0.25, 0.25

        self.daily_rider_accepts = torch.zeros(self.num_riders)
        self.daily_num_requests = torch.zeros(self.num_riders)

        self.alphas_commuters = torch.ones(int(self.num_riders * self.commuter_percentage * 2), )
        self.betas_commuters = torch.ones(int(self.num_riders * self.commuter_percentage * 2), )
        self.alphas_party_goers = torch.ones(int(self.num_riders * self.party_goer_percentage), ) * 0.75
        self.betas_party_goers = torch.ones(int(self.num_riders * self.party_goer_percentage), ) * 0.75
        self.alphas_sporadic = torch.ones(int(self.num_riders * self.sporadic_rider_percentage), ) * 0.75
        self.betas_sporadic = torch.ones(int(self.num_riders * self.sporadic_rider_percentage), ) * 0.75

        self.gamma_dist_commuters = torch.distributions.Gamma(self.alphas_commuters, self.betas_commuters)
        self.gamma_dist_party_goers = torch.distributions.Gamma(self.alphas_party_goers, self.betas_party_goers)
        self.gamma_dist_sporadic = torch.distributions.Gamma(self.alphas_sporadic, self.betas_sporadic)

        #make the lambdas of Poisson sampled from Gamma distributions as conjugate priors for Bayesian updates
        # self.lambda_riders = torch.cat([
        #     self.gamma_dist_commuters.sample(),
        #     self.gamma_dist_party_goers.sample(),
        #     self.gamma_dist_sporadic.sample()
        # ])
        self.lambda_riders = torch.cat(
        [
            torch.ones(int(self.num_riders * self.commuter_percentage * 2), ),
            torch.ones(int(self.num_riders * self.party_goer_percentage), ) * 0.75,
            torch.ones(int(self.num_riders * self.sporadic_rider_percentage), ) * 0.75
        ]
        )

        # self.log_normal_mu = 60
        # self.log_normal_sigma = 0.5
        
        # for idle duration
        self.exp_shift = 2 * 60

        self.daily_driver_rejects = torch.zeros(self.num_drivers)

        self.subgrid_bounds = torch.zeros(self.num_subgrids, 2, 2)
        for i in range(self.num_subgrids_per_dim):
            for j in range(self.num_subgrids_per_dim):
                subgrid_index = i * self.num_subgrids_per_dim + j
                self.subgrid_bounds[subgrid_index, 0] = torch.tensor([i * self.subgrid_size, (i + 1) * self.subgrid_size])
                self.subgrid_bounds[subgrid_index, 1] = torch.tensor([j * self.subgrid_size, (j + 1) * self.subgrid_size])

        self.D_Requests = None
        self.S_Drivers = None

        self.exposed_prices = []
        self.exposed_prices_lst = []

        #self.busy_drivers = {} # idx of drivers on a trip(maybe add sub-block of the driver): trip ending timestamp
        #no need to do so as we update each driver's idle_start_time based on trips

    def simulate_demand(self):
        """
        simulate requests of riders (demands - D) every day
        return:
        D[:,0] gives all the timestamps(from 0 to 24*60-1) of the ride_requests, 
        D[:,1] gives the x_start, D[:,2] gives the y_start, D[:,3] gives the x_end and D[:,4] gives the y_end
        D[:, 5] gives each request's rider_idx which could be used to indicate the rider type.
        D[:,6] gives the sub-block id for each request trip start position
        D[:,7] gives the sub-block id for each request trip end position
        """
        # each rider's number of request on this day e.g. [0., 2., 1.]
        self.daily_num_requests = torch.poisson(self.lambda_riders)
        # sequential ID for total number of rides
        request_indices = torch.arange(self.daily_num_requests.sum().int())

        # rider index repeats each rider's sequential ID according to the number of requests for that rider.
        rider_indices = torch.repeat_interleave(torch.arange(self.num_riders), self.daily_num_requests.int())

        
        request_times = torch.zeros(len(request_indices))

        request_times[self.mask1[rider_indices]] = torch.normal(self.request_times_means[0], self.request_times_stds[0], size=(self.mask1[rider_indices].sum(),))
        request_times[self.mask2[rider_indices]] = torch.normal(self.request_times_means[1], self.request_times_stds[1], size=(self.mask2[rider_indices].sum(),))
        request_times[self.mask3[rider_indices]] = torch.normal(self.request_times_means[2], self.request_times_stds[2], size=(self.mask3[rider_indices].sum(),))

        # Resample values outside the valid range, could be costly, assume always resample from commuter type 1 for simplicity
        invalid_mask = (request_times < 0) | (request_times >= 24 * 60)
        while invalid_mask.any():
            request_times[invalid_mask] = torch.normal(self.request_times_means[0], self.request_times_stds[0], size=(invalid_mask.sum(),))
            invalid_mask = (request_times < 0) | (request_times >= 24 * 60)

        # Fill in random values for the remaining indices
        remaining_mask = ~(self.mask1[rider_indices] | self.mask2[rider_indices] | self.mask3[rider_indices])
        request_times[remaining_mask] = torch.randint(0, 24 * 60, size=(remaining_mask.sum(),), dtype=torch.float)


        start_locations = torch.rand(len(request_indices), 2)
        end_locations = torch.rand(len(request_indices), 2)

        self.D_Requests = torch.cat((request_times.unsqueeze(1), start_locations, end_locations
                                 , rider_indices.unsqueeze(1)), 1)
        
        self.D_Requests = torch.concat((self.D_Requests
                                , self.get_subblock_index(self.D_Requests[:, 1], self.D_Requests[:, 2]).unsqueeze(1)
                                , self.get_subblock_index(self.D_Requests[:, 3], self.D_Requests[:, 4]).unsqueeze(1))
                                , 1)

        return self.D_Requests
    

    def simulate_supply(self):
        """
        simulate distribution of drivers (Supplies - S) every start of the day
        return:
        S[:,0] gives idle start time timestamps(from 0 to 24*60-1) which the driver begins to be matched to ride requests, 
        S[:,1] gives the idle duration(number of minutes the driver can be matched), 
        S[:,2] gives the x idle start position, S[:,3] gives the y idle start position
        S[:,4] gives the driver index(0 to 99)
        S[:,5] gives the sub-block id for each idle start position
        """

        # Assign each driver to a subgrid
        driver_subgrids = torch.randint(0, self.num_subgrids, (self.num_drivers,))

        # Initialize the driver positions randomly within their assigned subgrids
        driver_positions = torch.zeros(self.num_drivers, 2)

        for i in range(self.num_subgrids):
            mask = driver_subgrids == i
            x_min, x_max = self.subgrid_bounds[i, 0]
            y_min, y_max = self.subgrid_bounds[i, 1]
            driver_positions[mask, 0] = torch.rand(mask.sum()) * (x_max - x_min) + x_min
            driver_positions[mask, 1] = torch.rand(mask.sum()) * (y_max - y_min) + y_min

        # Define the probability distribution for driver idle times
        daytime_prob = 0.8  # Probability of drivers working during daytime hours
        nighttime_prob = 1 - daytime_prob
        daytime_hours = torch.arange(7, 23)  # Daytime hours (7 AM to 10 PM)
        nighttime_hours = torch.cat((torch.arange(0, 7), torch.arange(23, 24)))  # Nighttime hours (12 AM to 7 AM and 11 PM to 12 AM)
        idle_time_probs = torch.zeros(24)
        idle_time_probs[daytime_hours] = daytime_prob
        idle_time_probs[nighttime_hours] = nighttime_prob
        idle_time_probs /= idle_time_probs.sum()

        idle_starttime = torch.multinomial(idle_time_probs, self.num_drivers, replacement=True) * 60

        exponential_dist = torch.distributions.exponential.Exponential(torch.tensor(1.0 / self.mean_idle_time))

        # don't do resampling for now to save runtime, shift the exp var to have a minimum value
        idle_duration = (exponential_dist.sample((self.num_drivers,)) + self.exp_shift).clamp(min=1
                                                                                            , max=24 * 60)
        idle_duration = idle_duration.int().long()

        # Create the log-normal distribution
        #log_normal_dist = torch.distributions.log_normal.LogNormal(self.log_normal_mu, self.log_normal_sigma)

        # Sample durations from the log-normal distribution
        #idle_duration = log_normal_dist.sample((self.num_drivers,)).clamp(min=2*60, max=24*60).int().long()
        

        driver_indices = torch.arange(self.num_drivers)

        self.S_Drivers = torch.cat((idle_starttime.unsqueeze(1), idle_duration.unsqueeze(1)
                                  , driver_positions, driver_indices.unsqueeze(1)), 1)
        
        self.S_Drivers = torch.concat((self.S_Drivers
                                 , self.get_subblock_index(self.S_Drivers[:, 2], self.S_Drivers[:, 3]).unsqueeze(1))
                                 , 1)


        #just throw away those trips not finished by the end of the day
        return self.S_Drivers
    

    def get_subblock_index(self, x, y):
        # Find the subgrid indices along each dimension
        subgrid_x = torch.div(x, self.subgrid_size).int()
        subgrid_y = torch.div(y, self.subgrid_size).int()

        # Calculate the subgrid_index
        subgrid_index = subgrid_x * self.num_subgrids_per_dim + subgrid_y

        return subgrid_index
    
    def estimate_trip_distance_duration(self, pickup_dropoff_locations, driver_speed=0.5):
        # driver_speed is calculated by 30miles/hour(1mile/min), and 0.1 in the grid means 1 mile
        # Estimate the trip duration based on the pickup location and dropoff location
        # Return the estimated duration in seconds
        start_x, start_y, end_x, end_y = pickup_dropoff_locations[0], pickup_dropoff_locations[1], \
            pickup_dropoff_locations[2], pickup_dropoff_locations[3]
        euclidean_distance = np.sqrt((start_x - end_x)**2 + (start_y - end_y)**2) * 10
        trip_duration = np.round(euclidean_distance / driver_speed)

        return trip_duration, euclidean_distance

    def request_driver_matching(self, verbose=0, file_prefix=''):
        """
        match requests on drivers for every match_interval_time(30mins) and for each sub-block id
        self.S_Drivers: (num_drivers) * (idle_start_time_timestamps, idle_duration, idle_start_x, idle_start_y, 
                                        driver_idx, idle_start_subblock_id)

        self.D_Requests: (num_requests) * (request_timestamps, req_start_x, req_start_y, req_end_x, req_end_y, 
                                        rider_idx, req_start_subblock_id, req_end_subblock_id)
        """
        if verbose:
            logger = utils.create_logger(self.current_week, file_prefix, self.log_save_path)
        # TODO - figure out how to speed up, may do parallelization on the sub-blocks
        # enable tqdm for debugging only
        for interval_idx, match_interval in enumerate(range(0, 24*60-1, self.match_interval_time)):
            for square_index in range(self.num_subgrids_per_dim ** 2): 
                #replace idle_status_mask with checking on the idle_time
                idle_time_mask_left = self.S_Drivers[:, 0]<(interval_idx+1)*self.match_interval_time
                idle_time_mask_right = interval_idx*self.match_interval_time<=self.S_Drivers[:, 0]+self.S_Drivers[:, 1]
                idle_location_mask = self.S_Drivers[:, 5]==square_index

                drivers_subblock = self.S_Drivers[idle_time_mask_left & idle_time_mask_right & idle_location_mask]
                #print(f'len(drivers_subblock):{len(drivers_subblock)}')


                if len(drivers_subblock)==0:
                    #print(f'no idle driver in this sub-block:{square_index}')
                    continue
                else:
                    #print(f'at least one idle driver in this sub-block:{square_index}')
                    pass


                request_time_mask_left = interval_idx*self.match_interval_time<=self.D_Requests[:, 0]
                request_time_mask_right = self.D_Requests[:, 0]<(interval_idx+1)*self.match_interval_time
                request_location_mask = self.D_Requests[:, 6]==square_index

                riders_subblock = self.D_Requests[request_time_mask_left & request_time_mask_right \
                                                & request_location_mask]
                
                if len(riders_subblock)==0:
                    #print(f'no idle driver in this sub-block:{square_index}')
                    continue
                else:
                    #print(f'at least one rider request in this sub-block:{square_index}')
                    pass

                #have to iterate through every rider in the given time interval and the sub-block is both rider and driver are valid

                for valid_request_id in range(riders_subblock.shape[0]):
                    valid_driver = drivers_subblock.shape[0]
                    
                    selected_driver = drivers_subblock[torch.randint(0, valid_driver, (1,)).item(), :][0] if valid_driver>1 else drivers_subblock[0]
                    if valid_driver==0 or selected_driver.dim()==0:
                        #print('no more valid driver to be matched!')
                        break

                    ride_minutes, ride_miles = self.estimate_trip_distance_duration(riders_subblock[valid_request_id][1:5])
                    price_of_ride = self.pricing_params[0] + self.pricing_params[1] * ride_minutes + self.pricing_params[2] * ride_miles
                    
                    #normalize price_of_ride by ride_miles to ensure a conditional elasticity for short-term demand
                    normalized_price = price_of_ride/ride_miles if ride_miles>1 else price_of_ride

                    rider_acceptance_prob = float(torch.sigmoid(self.a_r + self.b_r * normalized_price))
                    driver_acceptance_prob = float(torch.sigmoid(self.a_d + self.b_d * normalized_price))

                    self.exposed_prices.append({
                        'price_of_ride': round(float(price_of_ride), 4),
                        'distance_normalized_price': round(float(normalized_price), 4),
                        'trip_duration': round(float(ride_minutes), 4),
                        'rider_acceptance_prob': round(float(rider_acceptance_prob), 4),
                        'driver_acceptance_prob': round(float(driver_acceptance_prob), 4)
                    })

                    self.exposed_prices_lst.append(round(float(price_of_ride), 4))

                    # Determine if the ride is accepted by both rider and driver
                    rider_acceptance_generator = np.random.rand()
                    driver_acceptance_generator = np.random.rand()
                    rider_id = int(riders_subblock[valid_request_id][5])
                    driver_idx = int(selected_driver[4])

                    #the rider accepts the trip
                    if rider_acceptance_generator < rider_acceptance_prob:
                        
                        self.daily_rider_accepts[rider_id] += 1

                        #the driver accepts the trip, update the trip info
                        if driver_acceptance_generator < driver_acceptance_prob:
                            #update driver's idle sub-block id to the trip destination
                            self.S_Drivers[driver_idx][5]==riders_subblock[valid_request_id][7]

                            #update their idle_start_timestamp to request_time+trip_duration 
                            prev_idle_start_timestamp = self.S_Drivers[driver_idx][0]
                            self.S_Drivers[driver_idx][0] = riders_subblock[valid_request_id][0] + int(ride_minutes)
                            #update idle_duration to the original idle_duration minus how much time has passed since the previous idle_start_timestamp to new idle_start_timestamp, (no negative values)
                            self.S_Drivers[driver_idx][1] = max(0, self.S_Drivers[driver_idx][1] - (self.S_Drivers[driver_idx][0] - prev_idle_start_timestamp))

                            if verbose:
                                log_entry = {'current_day': self.current_day,'square_index': square_index, 'rider_id': rider_id, 'driver_idx': driver_idx, \
                                            'trip_start_timestamp': int(riders_subblock[valid_request_id][0]), 'trip_duration': round(float(ride_minutes), 4), \
                                            'ride_miles': round(float(ride_miles), 4), 'trip_end_timestamp': int(self.S_Drivers[driver_idx][0]), \
                                            'price_of_ride': round(float(price_of_ride), 4), 'distance_normalized_price': round(float(normalized_price), 4), \
                                            'rider_acceptance_prob': round(float(rider_acceptance_prob), 4), 'driver_acceptance_prob': round(float(driver_acceptance_prob), 4)}
                                logger.debug(json.dumps(log_entry))
                            
                            #remove the current busy driver in drivers_subblock
                            drivers_subblock = drivers_subblock[drivers_subblock[:, 4]!=driver_idx]
                            if len(drivers_subblock)==0:
                                #no more valid drivers, exit the loop right away
                                break

                        #the driver rejects the trip
                        else:
                            #driver_acceptance_generator >= driver_acceptance_prob
                            #TODO - not sure how we will use it in later days
                            self.daily_driver_rejects[driver_idx] += 1

                    #the rider rejects the trip, do nothing as we update the accepts only
                    #else:


    def update_gamma_distns(self):
        """
        daily update on the alphas and betas in the gamma priors for poisson based on number of accepts and number of requests
        """
        self.alphas_commuters += self.daily_rider_accepts[self.mask1 | self.mask2]
        self.betas_commuters += self.daily_num_requests[self.mask1 | self.mask2]

        self.alphas_party_goers += self.daily_rider_accepts[self.mask3]
        self.betas_party_goers += self.daily_num_requests[self.mask3]

        remaining_mask = ~(self.mask1 | self.mask2 | self.mask3)
        self.alphas_sporadic += self.daily_rider_accepts[remaining_mask]
        self.betas_sporadic += self.daily_num_requests[remaining_mask]

        self.gamma_dist_commuters = torch.distributions.Gamma(self.alphas_commuters, self.betas_commuters)
        self.gamma_dist_party_goers = torch.distributions.Gamma(self.alphas_party_goers, self.betas_party_goers)
        self.gamma_dist_sporadic = torch.distributions.Gamma(self.alphas_sporadic, self.betas_sporadic)

        self.current_day += 1

    def update_lambda_longterm_elasticity(self):
        """
        daily update on the lambda based on long-term rider's price elasticity
        """
        exposed_prices_arr = np.array(self.exposed_prices_lst)
        daily_avg_exposed_price = np.mean(exposed_prices_arr)
        #updated_lambda = np.exp(self.a_lambda * daily_avg_exposed_price + self.b_lambda)
        updated_lambda = self.a_lambda * (daily_avg_exposed_price - self.b_lambda)
        #print(f'daily_avg_exposed_price:{daily_avg_exposed_price}')
        #print(f'updated_lambda:{updated_lambda}')
        self.lambda_riders = torch.clamp(self.lambda_riders + updated_lambda, min=0)

            