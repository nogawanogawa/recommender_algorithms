import urllib
from zipfile import ZipFile
import urllib.request
import pathlib
import os

def prepare_dataset():
    if pathlib.Path('ml-100k').exists():
        return True
    else:
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        urllib.request.urlretrieve(url, 'ml-100k.zip')

        with ZipFile('ml-100k.zip', 'r') as zip:
            zip.printdir()
            zip.extractall()
        os.remove("ml-100k.zip")
        return True

if __name__ == '__main__':
    result = prepare_dataset()
    if not result:
        print("ERROR")
    else:
        print("SUCCESS")
