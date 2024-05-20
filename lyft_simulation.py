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

        self.riders = None
        self.drivers = None

    def simulate_riders(self):
        """
        simulate requests of riders every day
        return:
        D[:,0] gives all the timestamps(from 0 to 24*60-1) of the ride_requests, 
        D[:,1] gives the x_start, D[:,2] gives the y_start, D[:,3] gives the x_end and D[:,4] gives the y_end
        """
        daily_num_requests = torch.poisson(self.lambda_riders).clamp(max=self.daily_max_requests)
        request_indices = torch.arange(daily_num_requests.sum().int())

        # rider index for each request
        rider_indices = torch.repeat_interleave(torch.arange(self.num_riders), daily_num_requests.int())
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

        riders = torch.cat((request_times, start_locations, end_locations), 1)

        return riders