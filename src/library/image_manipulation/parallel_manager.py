"""This module helps use multiple processes (cores)
to process multiple images simultaneously.
"""
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import wait, ThreadPoolExecutor

from library.utilities.utilities_process import get_hostname


class ParallelManager:
    """Methods to support processing any part of pipeline (discreet function) using multiple cores
    """
    def __init__(self, debug: bool = False):
        """Initialize the ParallelManager with debug mode.

        :param debug: If True, runs in debug mode with single-threaded execution.
        """
        self.debug = debug

    def get_nworkers(self):
        """Get the number of cores to use per workstation. The same
        number is used for both downsampled and full resolution images.
        There is little point in setting two different levels in one host.
        Much effort was used to set these numbers.
        The number must be 2 or greater
        """
        usecpus = 4
        cpus = {}
        cpus['mothra'] = 2
        cpus['tobor'] = 6
        cpus['muralis'] = 10
        cpus['basalis'] = 4
        cpus['ratto'] = 4
        hostname = get_hostname()
        if hostname in cpus.keys():
            usecpus = cpus[hostname]
        return usecpus

    def run_commands_concurrently(self, function, file_keys, workers):
        """This method uses the ProcessPoolExecutor library to run
        multiple processes at the same time. It also has a debug option.
        This is helpful to show errors on stdout. 

        :param function: the function to run
        :param file_keys: tuple of file information
        :param workers: integer number of workers to use
        """
        
        if self.debug:
            for file_key in sorted(file_keys):
                function(file_key)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                executor.map(function, sorted(file_keys))
                executor.shutdown(wait=True)


    def run_commands_with_threads(self, function, file_keys, workers):
        """This method uses the ThreadPoolExecutor library to run
        multiple processes at the same time. It also has a debug option.
        This is helpful to show errors on stdout. 

        :param function: the function to run
        :param file_keys: tuple of file information
        :param workers: integer number of workers to use
        """
        
        if self.debug:
            for file_key in sorted(file_keys):
                function(file_key)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                #executor.map(function, sorted(file_keys))
                #executor.shutdown(wait=True)
                futures = [executor.submit(function, file_key) for file_key in sorted(file_keys)]
                # wait for all tasks to complete
                wait(futures)
