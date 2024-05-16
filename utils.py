import uuid
import numpy as np







def generate_rider_id(self):
    """
    Generate a unique rider ID
    128-bit numbers typically presented in hexadecimal form, resulting in a 32-character string
    """
    return uuid.uuid4()

def generate_driver_id(self):
    """
    Generate a unique driver ID
    """
    return uuid.uuid4()

def generate_trip_id(self):
    """
    Generate a unique trip ID
    """
    return uuid.uuid4()

# TODO - put irregularity (e.g. city and airport) on the grid
def generate_random_location(self):
    """
    Generate a random location (i. e., x, y coordinates each from (0, 1))
    """
    return (np.random.rand(), np.random.rand())

def create_probability_density(city_locations, airport_locations):
    """
    returns a density function with a list of city and airport as inputs
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
    use rejection sampling to generate random a location x & y based on the given density function
    """
    while True:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        density = probability_density(x, y)
        if np.random.uniform(0, 1) < density / density_divider:  # Adjust the acceptance threshold as needed
            return x, y