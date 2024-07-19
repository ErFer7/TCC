"""
Dataset.
"""

from os import environ
from os.path import join
from json import load
from zipfile import ZipFile

import pandas as pd

KAGGLE_CREDENTIALS_PATH = join('..' ,'.secrets', 'kaggle.json')
DATA_PATH = join('..', '.data')

with open(KAGGLE_CREDENTIALS_PATH, 'r', encoding='utf-8') as file:
    json = load(file)
    environ['KAGGLE_USERNAME'] = json['username']
    environ['KAGGLE_KEY'] = json['key']

import kaggle

kaggle.api.authenticate()
kaggle.api.competition_download_file('siim-isic-melanoma-classification',
                                     'train.csv',
                                     path=DATA_PATH,
                                     quiet=False,
                                     force=False)

with ZipFile(join(DATA_PATH, 'train.csv.zip'), 'r') as file:
    file.extractall(join(DATA_PATH, 'extracted'))

train_data = pd.read_csv(join(DATA_PATH, 'extracted', 'train.csv'))
