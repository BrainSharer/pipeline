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

    def __init__(self, LOGFILE_PATH):
        """
        -SET CONFIG FOR LOGGING TO FILE; ABILITY TO OUTPUT TO STD OUTPUT AND FILE

        """

        LOGFILE = os.path.join(LOGFILE_PATH, "pipeline-process.log")

        # SET ENV FOR OTHER [NON-PIPELINE] MODULES
        if environ.get("LOGFILE_PATH") is None:
            os.environ["LOGFILE_PATH"] = LOGFILE_PATH

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG) #THRESHOLD FOR LOGGER
        formatter = logging.Formatter("%(message)s")

        # 'FOR LOOP' REMOVES DUAL LOGGING TO CONSOLE + FILE
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Create file handler for INFO and up (INFO, WARNING, ERROR, CRITICAL)
        fh = logging.FileHandler(LOGFILE)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # CONSOLE + LOG FILE
        # logger = logging.getLogger(__name__)
        # logger.setLevel(logging.DEBUG)
        # fh = logging.FileHandler(LOGFILE)
        # formatter = logging.Formatter("%(message)s")
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)

        self.filelogger = logger


    def logevent(self, msg: str):
        '''
        Implements output to terminal if debug is set to True (similar to linux command: tee)
        
        :param msg: accepts string comment that gets inserted into file log
        :type msg: str
        :return: timestamp of event is returned [unclear if used as of 4-NO-2022]
        '''
        timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        self.filelogger.info(f"{timestamp} - {msg}")
        if self.debug:
            print(f"{timestamp} - {msg}")
        return timestamp