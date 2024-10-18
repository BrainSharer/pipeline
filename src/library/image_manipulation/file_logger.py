"""Writes logging and debugging info to  file on birdstore.
"""

import os
from os import environ
import logging
from datetime import datetime


class FileLogger:
    """This class defines the file logging mechanism
    the first instance of FileLogger class defines default log file name and complete path 'LOGFILE_PATH'
    The full path is passed during application execution (i.e., running the pre-processing pipeline) and sets an
    environment variable for future file logging

    Optional configuration (defined in __init__) provide for concurrent output to file and console [currently
    only file output]

    Single method [outside of __init__] in class accepts log message as argument, creates current timestamp and saves to file
    """

    def __init__(self, LOGFILE_PATH, debug=False):
        """
        -SET CONFIG FOR LOGGING TO FILE; ABILITY TO OUTPUT TO STD OUTPUT AND FILE

        """

        LOGFILE = os.path.join(LOGFILE_PATH, "pipeline-process.log")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler
        LOGFILE = os.path.join(LOGFILE_PATH, "pipeline-process.log")
        file_handler = logging.FileHandler(LOGFILE)
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(file_handler)
        self.debug = debug

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)


    def logevent(self, msg: str):
        '''
        Implements output to terminal if debug is set to True (similar to linux command: tee)
        
        :param msg: accepts string comment that gets inserted into file log
        :type msg: str
        :return: timestamp of event is returned [unclear if used as of 4-NO-2022]
        '''
        timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{timestamp} - {msg}")
        if self.debug:
            print(f"{timestamp} - {msg}")
        return timestamp
    
