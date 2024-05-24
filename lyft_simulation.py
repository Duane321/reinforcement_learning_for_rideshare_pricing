import math
import itertools
from tqdm import tqdm
import json
import numpy as np
#import utils

import torch

class WeeklySimulation:


    def __init__(self, learning_rate, initial_pricing_params):
        self.lr = learning_rate # lr for SGD
        self.pricing_params = initial_pricing_params

        # Initialize acceptance probability parameters
        self.a_r = 1.0  # Rider acceptance probability parameter
        self.b_r = -0.1  # Rider acceptance probability parameter
        self.a_d = 1.0  # Driver acceptance probability parameter
        self.b_d = -0.05  # Driver acceptance probability parameter

        self.num_riders = 1000
        self.part_size = self.num_riders // 4
        self.daily_max_requests = 5
        
        self.num_drivers = 100
        self.num_subgrids = 16
        self.num_subgrids_per_dim = 4
        self.num_subgrids = self.num_subgrids_per_dim ** 2
        self.subgrid_size = 1.0 / self.num_subgrids_per_dim

        # on avg. a driver can work for 3hrs
        self.mean_idle_time = 180

        self.match_interval_time = 30

        #number of different distributions for request time
        self.num_components = 3 
        #480, 1080, 1320, total 1440, 0 is placeholder for uniform distribution
        self.request_times_means = torch.tensor([8 * 60, 18 * 60, 22 * 60, 0])
        self.request_times_stds = torch.tensor([60, 60, 60, 0])

        self.lambda_commuter, self.lambda_party_goer, self.lambda_sporadic = 1.5, 1, 1

        #there are two types of commuters(morning and evening),each with 25% probability
        self.commuter_percentage, self.party_goer_percentage, self.sporadic_rider_percentage = 0.25, 0.25, 0.25


        self.lambda_riders = torch.cat([
            torch.full((int(self.num_riders * self.commuter_percentage * 2),), self.lambda_commuter),
            torch.full((int(self.num_riders * self.party_goer_percentage),), self.lambda_party_goer),
            torch.full((int(self.num_riders * self.sporadic_rider_percentage),), self.lambda_sporadic)
        ])

        self.subgrid_bounds = torch.zeros(self.num_subgrids, 2, 2)
        for i in range(self.num_subgrids_per_dim):
            for j in range(self.num_subgrids_per_dim):
                subgrid_index = i * self.num_subgrids_per_dim + j
                self.subgrid_bounds[subgrid_index, 0] = torch.tensor([i * self.subgrid_size, (i + 1) * self.subgrid_size])
                self.subgrid_bounds[subgrid_index, 1] = torch.tensor([j * self.subgrid_size, (j + 1) * self.subgrid_size])

        self.riders = None
        self.drivers = None

        self.busy_drivers = {} # idx of drivers on a trip(maybe add sub-block of the driver): trip ending timestamp

    def simulate_riders(self):
        """
        simulate requests of riders (demands - D) every day
        return:
        D[:,0] gives all the timestamps(from 0 to 24*60-1) of the ride_requests, 
        D[:,1] gives the x_start, D[:,2] gives the y_start, D[:,3] gives the x_end and D[:,4] gives the y_end
        D[:, 5] gives each request's rider_idx which could be used to indicate the rider type.
        """
        # each rider's number of request on this day e.g. [0., 2., 1.]
        daily_num_requests = torch.poisson(self.lambda_riders).clamp(max=self.daily_max_requests)
        # sequential ID for total number of rides
        request_indices = torch.arange(daily_num_requests.sum().int())

        # rider index repeats each rider's sequential ID according to the number of requests for that rider.
        rider_indices = torch.repeat_interleave(torch.arange(self.num_riders), daily_num_requests.int())

        # sequential ID for total number of riders
        indices = torch.arange(self.num_riders)
        mask1 = indices < self.part_size
        mask2 = (indices >= self.part_size) & (indices < 2 * self.part_size)
        mask3 = (indices >= 2 * self.part_size) & (indices < 3 * self.part_size)
        #mask4 = indices >= 3 * self.part_size

        request_times = torch.where(
            mask1[rider_indices],
            torch.normal(self.request_times_means[0], self.request_times_stds[0], size=(len(request_indices),)),
            torch.where(
                mask2[rider_indices],
                torch.normal(self.request_times_means[1], self.request_times_stds[1], size=(len(request_indices),)),
                torch.where(
                    mask3[rider_indices],
                    torch.normal(self.request_times_means[2], self.request_times_stds[2], size=(len(request_indices),)),
                    torch.randint(0, 24 * 60, size=(len(request_indices),), dtype=torch.float)
                )
            )
        ).clamp(min=0, max=24*60-1).unsqueeze(1)


        start_locations = torch.rand(len(request_indices), 2)
        end_locations = torch.rand(len(request_indices), 2)

        self.riders = torch.cat((request_times, start_locations, end_locations
                                 , rider_indices.unsqueeze(1)), 1)

        return self.riders
    

    def simulate_drivers(self):
        """
        simulate distribution of drivers (Supplies - S) every start of the day
        return:
        S[:,0] gives idle start time timestamps(from 0 to 24*60-1) which the driver begins to be matched to ride requests, 
        S[:,1] gives the idle duration(number of minutes the driver can be matched), 
        S[:,2] gives the x idle start position, S[:,3] gives the y idle start position
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
        idle_duration = exponential_dist.sample((self.num_drivers,)).clamp(min=1, max=24 * 60).int().long()

        driver_indices = torch.arange(self.num_drivers)

        self.drivers = torch.cat((idle_starttime.unsqueeze(1), idle_duration.unsqueeze(1)
                                  , driver_positions, driver_indices.unsqueeze(1)), 1)

        return self.drivers
    

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
        start_x, start_y, end_x, end_y = pickup_dropoff_locations[0], pickup_dropoff_locations[1], pickup_dropoff_locations[0], pickup_dropoff_locations[1]
        euclidean_distance = np.sqrt((start_x - end_x)**2 + (start_y - end_y)**2) * 10
        trip_duration = np.round(euclidean_distance / driver_speed)

        return trip_duration, euclidean_distance

    ### WIP
    def request_driver_matching(self):
        """
        self.drivers: (num_drivers) * (idle_start_time_timestamps, idle_duration, idle_start_x, idle_start_y, 
                                        driver_idx, idle_start_subblock_id, idle_status)

        self.riders: (num_requests) * (request_timestamps, req_start_x, req_start_y, req_end_x, req_end_y, 
                                        rider_idx, req_start_subblock_id, req_end_subblock_id)
        """
        for interval_idx, match_interval in enumerate(range(0, 24*60-1, self.match_interval_time)):
            for square_index in range(self.num_subgrids_per_dim ** 2): 
                idle_status_mask = self.drivers[:, 6]==0
                idle_time_mask_left = self.drivers[:, 0]<(interval_idx+1)*self.match_interval_time
                idle_time_mask_right = interval_idx*self.match_interval_time<=self.drivers[:, 0]+self.drivers[:, 1]
                idle_location_mask = self.drivers[:, 5]==square_index

                drivers_subblock = self.drivers[idle_status_mask & idle_time_mask_left & idle_time_mask_right & idle_location_mask]


                # if len(drivers_subblock)==0:
                #     #print(f'no idle driver in this sub-block:{square_index}')
                #     continue
                # else:
                #     print(f'at least one idle driver in this sub-block:{square_index}')


                request_time_mask_left = interval_idx*self.match_interval_time<=self.riders[:, 0]
                request_time_mask_right = self.riders[:, 0]<(interval_idx+1)*self.match_interval_time
                request_location_mask = self.riders[:, 6]==square_index

                riders_subblock = self.riders[request_time_mask_left & request_time_mask_right \
                                                & request_location_mask]
                
                if len(riders_subblock)==0:
                    #print(f'no idle driver in this sub-block:{square_index}')
                    continue
                else:
                    print(f'at least one rider request in this sub-block:{square_index}')

                #have to iterate through every rider in the given time interval and the sub-block is both rider and driver are valid

                for valid_request_id in range(riders_subblock.shape[0]):
                    valid_driver = drivers_subblock.shape[0]
                    selected_driver = drivers_subblock[torch.randint(0, valid_driver, (1,)).item()] if valid_driver>1 else drivers_subblock

                    ride_minutes, ride_miles = self.estimate_trip_distance_duration(riders_subblock[valid_request_id][1:5])
                    price_of_ride = self.pricing_params[0] + self.pricing_params[1] * ride_minutes + self.pricing_params[2] * ride_miles


                    rider_acceptance_prob = torch.sigmoid(self.a_r + self.b_r * price_of_ride)
                    driver_acceptance_prob = torch.sigmoid(self.a_d + self.b_d * price_of_ride)

                    # Determine if the ride is accepted by both rider and driver
                    rider_acceptance_generator = np.random.rand()
                    driver_acceptance_generator = np.random.rand()
                    if rider_acceptance_generator < rider_acceptance_prob and driver_acceptance_generator < driver_acceptance_prob:
                        driver_idx = selected_driver[4]
                        #set the idle status to busy
                        self.drivers[driver_idx][6]==1
                        #remove busy the driver in drivers_subblock
                        #no need to set idle_start time because we have the idle status
                        drivers_subblock = drivers_subblock[drivers_subblock[:, 5]!=1]
                        #update driver's idle sub-block id to the trip destination
                        self.drivers[driver_idx][5]==riders_subblock[valid_request_id][7]

                        #update their idle_start_timestamp to request_time+trip_duration 
                        prev_idle_start_timestamp = self.drivers[driver_idx][0]
                        self.drivers[driver_idx][0] = riders_subblock[valid_request_id][0] + ride_minutes
                        #update idle_duration to the original idle_duration minus how much time has passed since the previous idle_start_timestamp to new idle_start_timestamp, (no negative values)
                        self.drivers[driver_idx][1] = max(0, self.drivers[driver_idx][1] - self.drivers[driver_idx][0] - prev_idle_start_timestamp)


                        

                    elif rider_acceptance_generator >= rider_acceptance_prob:
                        #TODO - update rider_rejects vector
                        pass


                    elif driver_acceptance_generator >= driver_acceptance_prob:
                        #TODO - update driver_rejects vector
                        pass
                
            if len(riders_subblock)!=0 and len(drivers_subblock)!=0:
                #print(f'no idle driver in this sub-block:{square_index}')
                break
            