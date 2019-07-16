import os
import tarfile
from urllib.request import urlretrieve

class Datasets(object):
    """
    A utility class used to download the necessary dataset(s).

    """

    def __init__(self, DOWNLOAD_URL):
        self.DOWNLOAD_URL = DOWNLOAD_URL
        self.DOWNLOAD_DIR = '../data/'

    def download(self, file_name):
        if not os.path.isdir(self.DOWNLOAD_URL):
            os.makedirs(self.DOWNLOAD_URL)

        path = os.path.join(self.DOWNLOAD_DIR, file_name)
        urlretrieve(self.DOWNLOAD_URL, path)
        tgz_file = tarfile.open(path)
        tgz_file.extractall(path=self.DOWNLOAD_DIR)
        tgz_file.close()

        