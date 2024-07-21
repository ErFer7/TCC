"""
Dataset.
"""

from uuid import uuid4
from json import load, dump
from os import environ
from os.path import join
from zipfile import ZipFile

import pandas as pd


DATA_PATH = join('.data')
KAGGLE_CREDENTIALS_PATH = join('.secrets', 'kaggle.json')

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

training_data = pd.read_csv(join(DATA_PATH, 'extracted', 'train.csv'))

dataset = []

for _, row in training_data.iterrows():
    dataset.append({
        'id': str(uuid4()),
        'image': row['image_name'],
        'conversations': [
            {
                'from': 'human',
                'value': 'Is this a melanoma benign or malignant?'
            },
            {
                'from': 'gpt',
                'value': f'This melanoma is {row['benign_malignant']}'
            }
        ]
    })

with open(join(DATA_PATH, 'training_dataset.json'), 'w', encoding='utf-8') as file:
    dump(dataset, file, ensure_ascii=False, indent=4)
