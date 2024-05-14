import heapq
import random
import numpy as np
from collections import defaultdict, namedtuple

Event = namedtuple('Event', ['timestamp', 'type', 'data'])

class LyftSimulation:
    #simulate for a week
    def __init__(self, learning_rate, initial_pricing_params, num_timestamps=7*24*60):
        self.events = [] # min-heap, order by event timestamp
        self.riders = {} # hashmap, rider_id: cur_loc, des_loc, status(idle or on a trip), num_rejects_by_rider
        self.drivers = {} # hashmap, driver_id: cur_loc, status(idle or on a trip)
        self.trips = defaultdict(list) # hashmap,(start_timestamp, trip_id): rider_id, driver_id, pickup_location
        self.num_timestamps = num_timestamps  # simulate every minute in a week for now (7 days * 24 hours * 60 minutes)

        # Initialize pricing parameters and learning rate
        self.pricing_params = np.array(initial_pricing_params)  # [c, p_per_min, p_per_mile]
        self.learning_rate = learning_rate # lr for SGD
        
        # Initialize acceptance probability parameters
        self.a_r = 1.0  # Rider acceptance probability parameter
        self.b_r = -0.1  # Rider acceptance probability parameter
        self.a_d = 1.0  # Driver acceptance probability parameter
        self.b_d = -0.05  # Driver acceptance probability parameter

        self.num_riders = 1000
        self.rider_types = {
            'commuter': 0.5,
            'party_goer': 0.2,
            'sporadic': 0.3
        }
        self.rider_request_rates = {
            'commuter': 2,  # Average requests per week for commuters
            'party_goer': 1,  # Average requests per week for party-goers
            'sporadic': 0.5  # Average requests per week for sporadic riders
        }
        self.riders = self.initialize_riders()

    def initialize_riders(self):
        riders = {}
        for i in range(self.num_riders):
            rider_id = f"r_{i}"
            rider_type = self.sample_rider_type()
            riders[rider_id] = {
                'type': rider_type,
                'requests': self.generate_rider_requests(rider_type)
            }
        return riders
    

    def sample_rider_type(self):
        return random.choices(list(self.rider_types.keys()), weights=list(self.rider_types.values()))[0]
    
    def generate_rider_requests(self, rider_type):
        num_requests = np.random.poisson(self.rider_request_rates[rider_type])
        requests = []
        for _ in range(num_requests):
            request_time = self.generate_request_time(rider_type)
            requests.append(request_time)
        return requests
    
    def generate_request_time(self, rider_type):
        if rider_type == 'commuter':
            # Generate request time based on commuting hours
            peak_hours = [(6, 10), (16, 18)]  # Morning and evening peak hours
            if random.random() < 0.8:  # 80% chance of requesting during peak hours
                peak_hour = random.choice(peak_hours)
                hour = random.randint(peak_hour[0], peak_hour[1])
            else:
                hour = random.randint(0, 23)
            minute = random.randint(0, 59)
        else:
            # Generate request time uniformly across the week
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)

        # Convert hour and minute to timestamp
        timestamp = hour * 60 + minute
        return timestamp
    
    def add_event(self, timestamp, event_type, data):
        heapq.heappush(self.events, Event(timestamp, event_type, data))

    def process_events(self):
        for timestamp in range(self.num_timestamps):
            self.update_riders(timestamp)
		
	    # sample new drivers and left drivers then assign their locations
            self.update_drivers(timestamp)

            while self.events and self.events[0].timestamp <= timestamp:
                event = heapq.heappop(self.events)
                _, event_type, data = event

		
                if event_type == 'rider_request':
                    self.process_rider_request(timestamp, data)
                elif event_type == ['trip_start']:
                    self.process_trip_start(timestamp, data)
		        # optional 
                elif event_type == 'driver_available':
                    self.process_driver_available(timestamp, data)
                elif event_type == 'trip_end':

                    self.process_trip_end(timestamp, data)

    def generate_rider_requests(self):
        for rider_id, rider_data in self.riders.items():
            num_requests = np.random.poisson(self.rider_request_rates[rider_data['type']])
            for _ in range(num_requests):
                request_time = self.generate_request_time(rider_data['type'])
                pickup_location = self.generate_random_location()
                self.add_event(request_time, 'rider_request', (rider_id, pickup_location))

    def process_rider_request(self, timestamp, data):
        rider_id, pickup_location = data
        driver_id = self.find_a_driver(pickup_location)
        if driver_id:
            # Create a new trip
            trip_id = self.generate_trip_id()
            start_timestamp = timestamp
            self.trips[(start_timestamp, trip_id)] = {
                'rider_id': rider_id,
                'driver_id': driver_id,
                'pickup_location': pickup_location
            }
            # Update rider and driver statuses
            self.riders[rider_id]['status'] = 'on_trip'
            self.drivers[driver_id]['status'] = 'on_trip'
            # Add a trip end event to the queue
            trip_duration = self.estimate_trip_duration(pickup_location)
            self.add_event(timestamp + trip_duration, 'trip_end', (start_timestamp, trip_id))
        else:
            # No available drivers, add the rider request back to the queue
            self.add_event(timestamp + 60, 'rider_request', data)  # Retry after 60 seconds

    def process_trip_end(self, timestamp, data):
        start_timestamp, trip_id = data
        trip_data = self.trips.pop((start_timestamp, trip_id))
        rider_id, driver_id = trip_data['rider_id'], trip_data['driver_id']
        # Update rider and driver statuses
        self.riders[rider_id]['status'] = 'idle'
        self.drivers[driver_id]['status'] = 'available'

    def find_a_driver(self, pickup_location):
        # Use the geospatial index to find the nearest available driver
        # Return the driver ID if found, otherwise return None
        pass

    def estimate_trip_duration(self, pickup_location):
        # Estimate the trip duration based on the pickup location
        # Return the estimated duration in seconds
        pass

    def generate_rider_id(self):
        # Generate a unique rider ID
        pass

    def generate_driver_id(self):
        # Generate a unique driver ID
        pass

    def generate_trip_id(self):
        # Generate a unique trip ID
        pass

    def generate_random_location(self):
        # Generate a random location (e.g., x, y coordinates)
        pass

    
    


















if __name__=="__main__":

    # Create and run simulations for multiple weeks
    num_weeks = 4  # Number of weeks to simulate

    for week in range(num_weeks):
        print(f"Running simulation for week {week + 1}")
        simulation = LyftSimulation()
        simulation.run_simulation()