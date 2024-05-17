import math
import heapq
import itertools
import random
#since we use rejection sampling, the sampled locations won't be exactly the same for each round of simulation
random.seed(42)
from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict, namedtuple
import utils

#Event = namedtuple('Event', ['timestamp', 'type', 'data'])

class Event:
    def __init__(self, timestamp, event_type, data):
        self.timestamp = timestamp
        self.event_type = event_type
        self.data = data
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, event_dict):
        return cls(event_dict["timestamp"], event_dict["event_type"], event_dict["data"])

#x, y, radius, density for each location
city_locations = [(0.1, 0.1, 0.2, 5.0), (0.5, 0.9, 0.2, 5.0)]
airport_locations = [(0.9, 0.1, 0.05, 10.0)]
probability_density = utils.create_probability_density(city_locations, airport_locations)


class LyftSimulation:
    #simulate for a week
    def __init__(self, learning_rate, initial_pricing_params, num_timestamps=7*24*60):
        self.events = [] # min-heap, order by event timestamp
        self.riders = {} # hashmap, rider_id: rider_type, requests, num_rejects_by_rider
        self.drivers = {} # hashmap, driver_id: cur_loc, status(idle or on a trip)
        self.trips = defaultdict(list) # hashmap, (start_timestamp, trip_id): rider_id, driver_id, pickup_location
        self.num_timestamps = num_timestamps  # simulate every minute in a week for now (7 days * 24 hours * 60 minutes)

        # Initialize pricing parameters and learning rate
        self.pricing_params = np.array(initial_pricing_params)  # [c, p_per_min, p_per_mile]
        self.learning_rate = learning_rate # lr for SGD
        
        # Initialize acceptance probability parameters
        self.a_r = 1.0  # Rider acceptance probability parameter
        self.b_r = -0.1  # Rider acceptance probability parameter
        self.a_d = 1.0  # Driver acceptance probability parameter
        self.b_d = -0.05  # Driver acceptance probability parameter

        self.num_drivers = 100
        

        self.num_riders = 1000
        

        self.rider_types = {
            'commuter': 0.5,
            'party_goer': 0.2,
            'sporadic': 0.3
        }
        self.rider_request_rates = {
            'commuter': 5,  # Average requests per week for commuters
            'party_goer': 2,  # Average requests per week for party-goers
            'sporadic': 1  # Average requests per week for sporadic riders
        }

        self.rider_requests_cap = {
            'commuter': 10,  # Average requests per week for commuters
            'party_goer': 7,  # Average requests per week for party-goers
            'sporadic': 3  # Average requests per week for sporadic riders
        }
        

    def initialize_drivers(self):
        for i in range(self.num_drivers):
            driver_id = utils.generate_driver_id()
            self.drivers[driver_id] = {
                # TODO - Drivers sit in a subgrid (e.g. x=[.1, .2], y=[.4,.5]) and wait for ride requests.
                'cur_loc': utils.generate_random_location(probability_density),
                # 0 for idle, 1 for busy
                'status': 0
            }
        #return self.drivers
    
    # def set_drivers(self, from_filename=None):
    #     if not from_filename:
    #         self.drivers = self.initialize_drivers()
    #     else:
    #         self.drivers = utils.read_json(from_filename)

    def initialize_riders(self):
        for _ in range(self.num_riders):
            rider_id = utils.generate_rider_id()
            rider_type = self.sample_rider_type()
            work_address = utils.generate_random_location(probability_density) if rider_type=='commuter' else None
            self.riders[rider_id] = {
                'type': rider_type,
                'num_rejects_by_rider': 0,
                'home_address': utils.generate_random_location(probability_density),
                'work_address': work_address,
                'requests': []
            }


        #return self.riders
    
    def add_requests(self):
        #need to setup requests with addresses when every rider's id and address is created
        rider_id_lst = list(self.riders.keys())
        for rider_id in tqdm(rider_id_lst):
            #print(f'rider_id:{rider_id}')
            rider_type = self.riders[rider_id]['type']
            self.riders[rider_id]['requests'] = self.generate_rider_requests(rider_id, rider_type)

        #return self.riders
    
    # def set_riders(self, from_filename=None):
    #     if not from_filename:
    #         self.riders = self.initialize_riders()
    #     else:
    #         self.riders = utils.read_json(from_filename)
    
    # TODO - vectorize by generating a (1, 1000) vector with values 0-1
    def sample_rider_type(self):
        return random.choices(list(self.rider_types.keys()), weights=list(self.rider_types.values()))[0]
    
    def add_event(self, timestamp, event_type, data):
        heapq.heappush(self.events, Event(timestamp, event_type, data))

    #add event on a file
    #def add_event_backtest(self, timestamp, event_type, data):

    
    def generate_rider_requests(self, rider_id, rider_type):
        """
        given a list of requests with day, hour, trip_type(leave, return),
        generate start and end locations for each trip request
        """
        #draw a sample from poisson distn (type: int)
        num_requests = np.random.poisson(self.rider_request_rates[rider_type])
        num_requests = min(num_requests, self.rider_requests_cap[rider_type])
        
        requests = []
        request_types = ['leave', 'return']

        if rider_type=='commuter':
            #random.choices(request_days_types, num_requests), sample with replacement from cartesian products of request_day * request_type
            request_days = list(range(1, 6))
            
            request_days_types = list(itertools.product(request_days, request_types))
            sampled_days_types = random.choices(request_days_types, k=num_requests)
            hours_dict = {
                'leave': [6, 7, 8, 9, 10],
                'return': [16, 17, 18]
            }

            for request_day, request_type in sampled_days_types:
                pickup_location, dropoff_location = [], []
                hour = random.choice(hours_dict[request_type])
                minute = random.randint(0, 59)
                request_time = (request_day-1) * 24 * 60 + hour * 60 + minute
                
                #print(f'ith request:{i}')
                #print(f'request_time, request_day, request_type:{(request_time, request_day, request_type)}')
                #request_type should be either 'leave' or 'return' for a 'commuter'
                if request_type=='leave':
                    pickup_location, dropoff_location = self.riders[rider_id]['home_address'], self.riders[rider_id]['work_address']
                elif request_type=='return':
                    pickup_location, dropoff_location = self.riders[rider_id]['work_address'], self.riders[rider_id]['home_address']
                
                requests.append([request_time, request_day, request_type, pickup_location, dropoff_location])
                self.add_event(request_time, 'rider_request', (rider_id, pickup_location, dropoff_location))

                if request_time>10080:
                    print(f'request_time, request_day, hour, minute:{request_time, request_day, hour, minute}')
        
        elif rider_type=='party_goer':
            request_days = list(range(1, 8))
            request_days_types = list(itertools.product(request_days, request_types))
            sampled_days_types = random.choices(request_days_types, k=num_requests)
            hours_dict = {
                'leave': [18, 19, 20, 21, 22, 23],
                'return': [0, 1, 2, 3, 4, 5]
            }

            others_addresses_lst = [v['home_address'] for k, v in self.riders.items() if k!=rider_id]
            #one random party address for each day in a week
            party_address = random.choices(others_addresses_lst, k=len(request_days))

            for request_day, request_type in sampled_days_types:
                pickup_location, dropoff_location = [], []
                hour = random.choice(hours_dict[request_type])
                minute = random.randint(0, 59)
                request_time = (request_day-1) * 24 * 60 + hour * 60 + minute
                
                #print(f'ith request:{i}')
                #print(f'request_time, request_day, request_type:{(request_time, request_day, request_type)}')
                #request_type should be either 'leave' or 'return' for a 'party_goer'
                if request_type=='leave':
                    pickup_location, dropoff_location = self.riders[rider_id]['home_address'], party_address[request_day-1]
                elif request_type=='return':
                    pickup_location, dropoff_location = party_address[request_day-1], self.riders[rider_id]['home_address']
            
                requests.append([request_time, request_day, request_type, pickup_location, dropoff_location])
                self.add_event(request_time, 'rider_request', (rider_id, pickup_location, dropoff_location))

                if request_time>10080:
                    print(f'request_time, request_day, hour, minute:{request_time, request_day, hour, minute}')

        else:
            for i in range(num_requests):
                pickup_location, dropoff_location = [], []
                random_walk_generator = np.random.rand()
                request_day = random.randint(1, 7) # sample a random day (both sides included)
                request_type = random.choice(request_types)


                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
                request_time = (request_day-1) * 24 * 60 + hour * 60 + minute
                
                #print(f'ith request:{i}')
                #print(f'request_time, request_day, request_type:{(request_time, request_day, request_type)}')
                if random_walk_generator < 0.5:
                    #50% of time the rider request a ride from two random locations(ignore request_type returned)
                    pickup_location, dropoff_location = utils.generate_random_location(probability_density), utils.generate_random_location(probability_density)

                else:
                    #25% of time the rider request a ride from home_address
                    if request_type=='leave':
                        pickup_location, dropoff_location = self.riders[rider_id]['home_address'], utils.generate_random_location(probability_density)
                    #25% of time the rider request a ride to home_address
                    elif request_type=='return':
                        pickup_location, dropoff_location = utils.generate_random_location(probability_density), self.riders[rider_id]['home_address']
                        
                requests.append([request_time, request_day, request_type, pickup_location, dropoff_location])
                self.add_event(request_time, 'rider_request', (rider_id, pickup_location, dropoff_location))

                if request_time>10080:
                    print(f'request_time, request_day, hour, minute:{request_time, request_day, hour, minute}')

        assert(num_requests==len(requests))

        

        return requests
    
 


    
    # TODO - sanity check on the generated rider requests(kde plots on each type of riders; a json file of all the rides)
    def generate_rider_requests_deprecated(self, rider_id, rider_type):
        #draw a sample from poisson distn (type: int)
        num_requests = np.random.poisson(self.rider_request_rates[rider_type])
        
        requests = []

        # TODO - sample both pickup_location and dropoff_location in city area for commuters
        if rider_type=='commuter':
            pickup_location, dropoff_location = utils.generate_random_location(), utils.generate_random_location()
        for _ in range(num_requests):
            request_time = self.generate_request_time(rider_type)
            requests.append(request_time)
            if rider_type!='commuter':
                pickup_location, dropoff_location = utils.generate_random_location(), utils.generate_random_location()
            self.add_event(request_time, 'rider_request', (rider_id, pickup_location, dropoff_location))
        return requests
    
    
    def generate_a_request_time_deprecated(self, rider_type):
        # DONE TODO - later we may fix each commuters ride and time in each week(only sample once per week instead of per day)
        # only sample once on each start of week(in initialize_riders)
        trip_type = ''
        if rider_type == 'commuter':
            request_day = random.randint(1, 5) # sample a random day during week days (both sides included)
            
            timerange_selector = np.random.rand()
            if timerange_selector < 0.5:
                peak_hours = [6, 7, 8, 9, 10] # morning peak hours
                trip_type='leave'
            else:
                peak_hours = [16, 17, 18] # evening peak hours
                trip_type='return'
            # Generate request time based on commuting hours
            hour = random.choice(peak_hours)
            #hour = random.randint(peak_hour[0], peak_hour[1]), both are O(1)
            minute = random.randint(0, 59)
        elif rider_type == 'party_goer':
            party_day = [1, 2, 3, 4] + [5, 6, 7] * 3 #put more weights on Fri. Sat. and Sun.
            request_day = random.choice(party_day) # sample a random day 
            timerange_selector = np.random.rand()
            if timerange_selector < 0.5:
                party_hours = [18, 19, 20, 21, 22, 23] 
                trip_type='leave'
            else:
                party_hours = [0, 1, 2, 3, 4, 5]
                trip_type='return'
            # Generate request time based on party ending hours
            hour = random.choice(party_hours)
            minute = random.randint(0, 59)
        else:
            request_day = random.randint(1, 7) # sample a random day (both sides included)
            # Generate request time uniformly across the week
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            # for simplicity, assume the sporadic rider leaves home from 6am to 3pm and returns otherwise
            if 6<=hour<=15:
                trip_type='leave'
            else:
                trip_type='return'

        # Convert hour and minute to timestamp
        timestamp = request_day * (hour * 60 + minute)


        return timestamp, request_day, trip_type

    def process_rider_request(self, timestamp, data, rider_reject_effect=1):
        """
        Handle trip matching, pricing and trip update
        """
        rider_id, pickup_location, dropoff_location = data
        # TODO - later add a pickup_location based matching function
        driver_id = self.find_a_driver()
        if driver_id:
            # Calculate ride details
            ride_minutes, ride_miles = self.estimate_trip_distance_duration(pickup_location, dropoff_location)
            
            # Prepare input features for the pricing model
            day, hour, minute = self.get_time_of_week(timestamp)
            model_input = np.array([pickup_location[0], pickup_location[1], day, hour, minute])
            
            # Calculate price_of_ride using the linear model # pricing_params=[c, p_per_min, p_per_mile]
            price_of_ride = np.dot(self.pricing_params, model_input)
            price_of_ride = self.pricing_params[0] + self.pricing_params[1] * ride_minutes + self.pricing_params[2] * ride_miles

            num_rejects_this_week = self.riders[rider_id]['num_rejects_by_rider']
            acceptance_prob_adjust = np.exp(-rider_reject_effect*num_rejects_this_week)
            
            # Calculate acceptance probabilities for riders and drivers
            rider_acceptance_prob = utils.sigmoid(self.a_r + self.b_r * price_of_ride) * acceptance_prob_adjust
            driver_acceptance_prob = utils.sigmoid(self.a_d + self.b_d * price_of_ride)

            # Determine if the ride is accepted by both rider and driver
            rider_acceptance_generator = np.random.rand()
            driver_acceptance_generator = np.random.rand()
            if rider_acceptance_generator < rider_acceptance_prob and driver_acceptance_generator < driver_acceptance_prob:
                # Create a new trip when both accepted
                trip_id = utils.generate_trip_id()
                start_timestamp = timestamp
                self.trips[(start_timestamp, trip_id)] = {
                    'rider_id': rider_id,
                    'driver_id': driver_id,
                    'pickup_location': pickup_location,
                    'dropoff_location': dropoff_location
                }
                # Update driver status to busy
                self.drivers[driver_id]['status'] = 1
                # Add a trip end event to the queue
                trip_duration = self.estimate_trip_duration(pickup_location, dropoff_location)
                self.add_event(timestamp + trip_duration, 'trip_end', (start_timestamp, trip_id))
            elif rider_acceptance_generator >= rider_acceptance_prob:
                # DONE TODO - change request parameters based on num_rejects_by_rider in a week
                self.riders[rider_id]['num_rejects_by_rider'] += 1
        # For simplicity, if the ride request is not accepted, the rider takes Uber instead, so no resending the requests

    def process_trip_end(self, data):
        start_timestamp, trip_id = data
        trip_data = self.trips.pop((start_timestamp, trip_id))
        rider_id, driver_id = trip_data['rider_id'], trip_data['driver_id']
        # Update driver status to idle
        self.drivers[driver_id]['status'] = 0

    def get_time_of_week(timestamp):
        #get time of a week based on timestamp

        day_of_week = timestamp//(60*24),
        mins_elapsed = day_of_week*60*24,
        mins_today = timestamp - mins_elapsed,
        hour_of_day = mins_today // 60

        return day_of_week, hour_of_day, mins_today

    def find_a_driver(self):
        # randomly assign an idle driver, could be optimized later

        idle_drive_lst = [driver_id for driver_id, val in self.drivers.items() if val['status'] == 0]

        return random.choice(idle_drive_lst)
        
        

    def estimate_trip_distance_duration(self, pickup_location, dropoff_location, driver_speed=0.5):
        # driver_speed is calculated by 30miles/hour(1mile/min), and 0.1 in the grid means 1 mile
        # Estimate the trip duration based on the pickup location and dropoff location
        # Return the estimated duration in seconds
        start_x, start_y, end_x, end_y = pickup_location[0], pickup_location[1], dropoff_location[0], dropoff_location[1]
        euclidean_distance = np.sqrt((start_x - end_x)**2 + (start_y - end_y)**2) * 10
        trip_duration = np.round(euclidean_distance / driver_speed)

        return trip_duration, euclidean_distance




    def run_simulation(self):
        for timestamp in range(self.num_timestamps):
            while self.events and self.events[0].timestamp <= timestamp:
                event = heapq.heappop(self.events)
                _, event_type, data = event

                if event_type == 'rider_request':
                    self.process_rider_request(timestamp, data)
                elif event_type == 'trip_start':
                    self.process_trip_start(timestamp, data)
                elif event_type == 'trip_end':
                    self.process_trip_end(data)


















if __name__=="__main__":

    # Create and run simulations for multiple weeks
    num_weeks = 4  # Number of weeks to simulate
    lr = 0.01
    #https://www.lyft.com/pricing/BKN
    #[c, p_per_min, p_per_mile]
    T0_pricing_params = (5, 0.78, 1.82)

    for week in range(num_weeks):
        print(f"Running simulation for week {week + 1}")
        simulation = LyftSimulation(lr, T0_pricing_params)
        simulation.run_simulation()