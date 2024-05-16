import uuid
import math
import json
import numpy as np

def generate_rider_id():
    """
    Generate a unique rider ID
    128-bit numbers typically presented in hexadecimal form, resulting in a 32-character string
    """
    return str(uuid.uuid4())

def generate_driver_id():
    """
    Generate a unique driver ID
    """
    return str(uuid.uuid4())

def generate_trip_id():
    """
    Generate a unique trip ID
    """
    return str(uuid.uuid4())

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def save_dict_to_json(d, filename):
    with open(filename, 'w') as fp:
        json.dump(d, fp, indent=4)

def read_json(filename):
    with open(filename, 'r') as fp:
        d = json.load(fp)

    return d

# TODO - put irregularity (e.g. city and airport) on the grid
def generate_random_location():
    """
    Generate a random location (i. e., x, y coordinates each from (0, 1))
    """
    return (np.random.rand(), np.random.rand())

def create_probability_density(city_locations, airport_locations):
    """
    Returns a density function with a list of city and airport as inputs
    """
    #x, y, radius, density
    #city_locations = [(0.1, 0.1, 0.2, 5.0)]
    #airport_locations = [(0.9, 0.1, 0.05, 10.0)]
    #probability_density = create_probability_density(city_locations, airport_locations)

    def probability_density(x, y):
        density = 1.0  # Default probability density

        # Increase density for city locations
        for city_location in city_locations:
            city_x, city_y, city_radius, city_density = city_location
            distance = np.sqrt((x - city_x)**2 + (y - city_y)**2)
            if distance <= city_radius:
                density += city_density  # Adjust the density increase as needed

        # Increase density for airport locations
        for airport_location in airport_locations:
            airport_x, airport_y, airport_radius, airport_density = airport_location
            distance = np.sqrt((x - airport_x)**2 + (y - airport_y)**2)
            if distance <= airport_radius:
                density += airport_density  # Adjust the density increase as needed

        return density
    
    #return the function itself to called
    return probability_density



def generate_random_location(probability_density, density_divider=20.0):
    """
    Use rejection sampling to generate random a location x & y based on the given density function
    """
    while True:
        x = np.random.rand()
        y = np.random.rand()
        density = probability_density(x, y)
        if np.random.rand() < density / density_divider:  # Adjust the acceptance threshold as needed
            return x, y