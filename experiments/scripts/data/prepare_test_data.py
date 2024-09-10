"""
Dataset.
"""

from json import dump
from random import shuffle
from os.path import join
from argparse import ArgumentParser

import pandas as pd

parser = ArgumentParser(description='Preparação de dados para testes')

parser.add_argument('samples', type=str, help='Número de amostras a serem testadas')

args = parser.parse_args()

training_data = pd.read_csv(join('data', 'extracted', 'train.csv'))

benign_samples = []
malignant_samples = []

for _, row in training_data.iterrows():
    image_file = f'{row["image_name"]}.jpg'

    if row['benign_malignant'] == 'benign':
        benign_samples.append({
            'image': image_file,
            'result': 'Benign'
        })
    else:
        malignant_samples.append({
            'image': image_file,
            'result': 'Malignant'
        })

dataset = []

if int(args.samples) > len(benign_samples) + len(malignant_samples):
    raise ValueError('Número de amostras maior que o número de amostras disponíveis')

while len(dataset) < int(args.samples):
    if len(benign_samples) > 0:
        dataset.append(benign_samples.pop(0))

    if len(malignant_samples) > 0:
        dataset.append(malignant_samples.pop(0))

shuffle(dataset)

with open(join('data', 'basic_test_dataset.json'), 'w', encoding='utf-8') as file:
    dump(dataset, file, ensure_ascii=False, indent=4)
