import math
import heapq
import random
import numpy as np
from collections import defaultdict, namedtuple
import utils

# TODO - put utility functions in a utility.py if there are too many of them
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

Event = namedtuple('Event', ['timestamp', 'type', 'data'])

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
        self.drivers = self.initialize_drivers()

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
        self.riders = self.initialize_riders()

    def initialize_drivers(self):
        drivers = {}
        for i in range(self.num_drivers):
            driver_id = utils.generate_driver_id()
            drivers[driver_id] = {
                'cur_loc': utils.generate_random_location(),
                # 0 for idle, 1 for busy
                'status': 0
            }
        return drivers

    def initialize_riders(self):
        riders = {}
        for i in range(self.num_riders):
            rider_id = utils.generate_rider_id()
            rider_type = self.sample_rider_type()
            riders[rider_id] = {
                'type': rider_type,
                'requests': self.generate_rider_requests(rider_id, rider_type),
                'num_rejects_by_rider': 0
            }
        return riders
    
    # TODO - vectorize by generating a (1, 1000) vector with values 0-1
    def sample_rider_type(self):
        return random.choices(list(self.rider_types.keys()), weights=list(self.rider_types.values()))[0]
    
    def add_event(self, timestamp, event_type, data):
        heapq.heappush(self.events, Event(timestamp, event_type, data))
    
    # TODO - sanity check on the generated rider requests(kde plots on each type of riders; a json file of all the rides)
    def generate_rider_requests(self, rider_id, rider_type):
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
    
    
    def generate_request_time(self, rider_type):
        # TODO - later we may fix each commuters ride and time in each week(only sample once per week instead of per day)
        if rider_type == 'commuter':
            request_day = random.randint(0, 4) # sample a random day during week days (both sides included)
            if random.random() < 0.8:  # 80% chance of requesting during peak hours
                # Generate request time based on commuting hours
                peak_hours = [6, 7, 8, 9, 10, 16, 17, 18]  # Morning and evening peak hours
                hour = random.choice(peak_hours)
                #hour = random.randint(peak_hour[0], peak_hour[1])
            else:
                hour = random.randint(0, 23)
            minute = random.randint(0, 59)
        elif rider_type == 'party_goer':
            party_day = [0, 1, 2, 3] + [4, 5, 6] * 3 #put more weights on Fri. Sat. and Sun.
            request_day = random.choice(party_day) # sample a random day 
            if random.random() < 0.8:  # 80% chance of requesting during party hours
                # Generate request time based on party ending hours
                party_hours = [22, 23, 0, 1, 2, 3, 4, 5]
                hour = random.choice(party_hours)
                #hour = random.randint(peak_hour[0], peak_hour[1])
            else:
                hour = random.randint(0, 23)
            minute = random.randint(0, 59)
        else:
            request_day = random.randint(0, 6) # sample a random day (both sides included)
            # Generate request time uniformly across the week
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)

        # Convert hour and minute to timestamp
        timestamp = request_day * (hour * 60 + minute)
        return timestamp

    def process_rider_request(self, timestamp, data):
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
            
            # Calculate acceptance probabilities for riders and drivers
            rider_acceptance_prob = sigmoid(self.a_r + self.b_r * price_of_ride)
            driver_acceptance_prob = sigmoid(self.a_d + self.b_d * price_of_ride)

            # Determine if the ride is accepted by both rider and driver
            if np.random.rand() < rider_acceptance_prob and np.random.rand() < driver_acceptance_prob:
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
            else:
                # TODO - change request parameters based on num_rejects_by_rider in a week
                self.riders[rider_id]['num_rejects_by_rider'] += 1
        # For simplicity, if the ride request is not accepted, the rider takes Uber instead

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
        # randomly assign an idle driver

        idle_drive_lst = [driver_id for driver_id, loc, status in self.drivers.items() if status == 0]

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