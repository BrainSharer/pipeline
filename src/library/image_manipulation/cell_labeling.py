import os

from library.utilities.utilities_process import SCALING_FACTOR, test_dir


class CellMaker:
    """Kui's cell labeler
    """

    def __init__(self):
        """Set up the class with the name of the file and the path to it's location.

        """



    def start_labels(self):
        """Get aligned images
        """
        INPUT = self.fileLocationManager.get_full_aligned(channel=self.channel)
        files = sorted(os.listdir(INPUT))
        for file in files:
            filepath = os.path.join(INPUT, file)
            ##### do stuff on each file here
            print(filepath)

