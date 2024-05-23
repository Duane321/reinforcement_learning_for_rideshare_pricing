import math
import itertools
from tqdm import tqdm
import json
import numpy as np
#import utils

import torch

class WeeklySimulation:


    def __init__(self, learning_rate, initial_pricing_params):
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

        riders = torch.cat((request_times, start_locations, end_locations, rider_indices.unsqueeze(1)), 1)

        return riders
    

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

        drivers = torch.cat((idle_starttime.unsqueeze(1), idle_duration.unsqueeze(1), driver_positions), 1)

        return drivers
    

    def get_subblock_index(self, x, y):
        # Find the subgrid indices along each dimension
        subgrid_x = torch.div(x, self.subgrid_size).int()
        subgrid_y = torch.div(y, self.subgrid_size).int()

        # Calculate the subgrid_index
        subgrid_index = subgrid_x * self.num_subgrids_per_dim + subgrid_y

        return subgrid_index