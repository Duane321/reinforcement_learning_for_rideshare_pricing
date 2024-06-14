import os
import logging
from pythonjsonlogger import jsonlogger

def create_logger(run_id, file_prefix):
    logger = logging.getLogger(f'logger_{run_id}')
    logger.setLevel(logging.DEBUG)
    
    # Create a unique filename for the log file
    filename = f'../data/10_weeks_a_r1.5_b_r-0.4/logfile_matched_trips_week_{run_id}_'+file_prefix+'.log'
    if not os.path.exists(filename):
        # Create the file if it doesn't exist
        open(filename, 'a').close()
    
    # Create a file handler with the filename
    file_handler = logging.FileHandler(filename, mode='a')  # write mode
    file_handler.setLevel(logging.DEBUG)
    
    # Create a JSON formatter
    formatter = logging.Formatter('%(message)s')  # No timestamp or logger name
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    # Ensure no propagation to the root logger
    logger.propagate = False
    
    return logger

def create_logger_custom(log_filename):
    logger = logging.getLogger(f'logger_{log_filename}')
    logger.setLevel(logging.DEBUG)
    
    # Create a unique filename for the log file
    filename = f'../data/{log_filename}.log'
    if not os.path.exists(filename):
        # Create the file if it doesn't exist
        open(filename, 'w').close()
    
    # Create a file handler with the filename
    file_handler = logging.FileHandler(filename, mode='a')  # write mode
    file_handler.setLevel(logging.DEBUG)
    
    # Create a JSON formatter
    formatter = logging.Formatter('%(message)s')  # No timestamp or logger name
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    # Ensure no propagation to the root logger
    logger.propagate = False
    
    return logger