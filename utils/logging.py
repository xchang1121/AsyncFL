# utils/logging.py

import logging
import sys
import os
from typing import Optional

def setup_logger(log_level: str = 'INFO', 
                   log_file: Optional[str] = None, 
                   name: str = 'FLSimLogger') -> logging.Logger:
    """
    Sets up a logger that outputs to both console and optionally a file.

    Args:
        log_level (str): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
        log_file (Optional[str]): Path to the log file. If None, only logs to console.
        name (str): Name of the logger instance.

    Returns:
        logging.Logger: The configured logger instance.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get the root logger or a specific logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define format
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                                   datefmt='%Y-%m-%d %H:%M:%S')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, mode='a') # Append mode
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    # Prevent propagation to root logger if using specific logger name
    # logger.propagate = False 

    return logger

# Example Usage (Typically called once at the start of main.py)
if __name__ == '__main__':
    
    # Example 1: Log INFO level to console only
    logger_console = setup_logger(log_level='INFO', name='ConsoleOnly')
    logger_console.debug("This debug message won't appear.")
    logger_console.info("This info message will appear on the console.")
    logger_console.warning("This is a warning.")

    print("\n" + "="*20 + "\n")

    # Example 2: Log DEBUG level to both console and file
    log_filename = 'experiment_run.log'
    logger_file = setup_logger(log_level='DEBUG', log_file=log_filename, name='FileLogger')
    logger_file.debug("This debug message will appear in console and file.")
    logger_file.info("This info message will appear too.")
    logger_file.error("This is an error message.")
    
    print(f"\nCheck the file '{log_filename}' for logged messages.")
    # Clean up dummy log file
    if os.path.exists(log_filename):
        # os.remove(log_filename) 
        # Keep the file for user to check after running this example
        pass