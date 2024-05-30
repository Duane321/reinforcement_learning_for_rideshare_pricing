import os
import logging
from pythonjsonlogger import jsonlogger



# # Function to create a new logger for each run
# def create_logger(run_id):
#     logger = logging.getLogger(f'logger_{run_id}')
#     logger.setLevel(logging.DEBUG)
    
#     # Create a file handler with a unique filename
#     filename = f'data/logfile_matched_trips_day_{run_id}.log'
#     if not os.path.exists(filename):
#         # Create the file if it doesn't exist
#         open(filename, 'w').close()

#     file_handler = logging.FileHandler(filename)
#     file_handler.setLevel(logging.DEBUG)
    
#     # Add the handler to the logger
#     logger.addHandler(file_handler)
    
#     # Ensure no propagation to the root logger
#     logger.propagate = False
    
#     return logger

def create_logger(run_id):
    logger = logging.getLogger(f'logger_{run_id}')
    logger.setLevel(logging.DEBUG)
    
    # Create a unique filename for the log file
    filename = f'data/logfile_matched_trips_day_{run_id}.log'
    if not os.path.exists(filename):
        # Create the file if it doesn't exist
        open(filename, 'w').close()
    
    # Create a file handler with the filename
    file_handler = logging.FileHandler(filename, mode='w')  # write mode
    file_handler.setLevel(logging.DEBUG)
    
    # Create a JSON formatter
    formatter = logging.Formatter('%(message)s')  # No timestamp or logger name
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    # Ensure no propagation to the root logger
    logger.propagate = False
    
    return logger