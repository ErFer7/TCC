"""
Dataset.
"""

from os import environ
from os.path import join
from json import load
from zipfile import ZipFile

import pandas as pd

with open(join('.secrets', 'kaggle.json'), 'r', encoding='utf-8') as file:
    json = load(file)
    environ['KAGGLE_USERNAME'] = json['username']
    environ['KAGGLE_KEY'] = json['key']

import kaggle

kaggle.api.authenticate()
kaggle.api.competition_download_file('siim-isic-melanoma-classification',
                                     'train.csv',
                                     path='.data',
                                     quiet=False,
                                     force=False)

with ZipFile(join('.data', 'train.csv.zip'), 'r') as file:
    file.extractall(join('.data', 'extracted'))

train_data = pd.read_csv(join('.data', 'extracted', 'train.csv'))
