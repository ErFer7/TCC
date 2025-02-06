'''
Preparação dos dados de treinamento.
'''

from re import search
from json import dump, load
from tqdm.notebook import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split

import pandas as pd

PROMPT = 'Classify the skin lesion in the image.'
ANSWER = 'The skin lesion in the image is {disease}.'


def format_data(selected_sample: dict) -> dict:
    '''
    Formata os dados.
    '''

    return {'messages': [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': PROMPT,
                }, {
                    'type': 'image',
                    'image': selected_sample['image'],
                }
            ],
        },
        {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': ANSWER.format(disease=selected_sample['dx'].replace('_', ' '))}],
        },
    ],
    }


def generate_training_messages(dataset: Dataset, size: int | None = None) -> list:
    '''
    Gera os dados de treinamento.
    '''

    if size is None:
        chunk_size = 1000
        training_messages = []

        for i in tqdm(range(0, len(dataset), chunk_size), desc='Processing chunks'):
            chunk = dataset[i:i + chunk_size]
            chunk_messages = [format_data(sample) for sample in chunk]
            training_messages.extend(chunk_messages)
    else:
        total_size = len(dataset)
        indices = range(total_size)

        labels = [sample['dx'] for sample in dataset]
        _, sampled_indices = train_test_split(
            indices,
            test_size=size / total_size,
            stratify=labels,
            random_state=42
        )

        # Process sampled indices in chunks
        training_messages = []
        for dx_index in tqdm(sampled_indices, desc='Processing samples'):
            sample = dataset[dx_index]
            training_messages.append(format_data(sample))

    return training_messages


def generate_test_samples(dataset: Dataset, size: int | None = None) -> list:
    '''
    Generates test data samples.
    Returns list of tuples containing (image, disease_label).
    '''

    if size is None:
        chunk_size = 1000
        test_samples = []

        for i in tqdm(range(0, len(dataset), chunk_size), desc='Processing chunks'):
            chunk = dataset[i:i + chunk_size]
            chunk_samples = [(sample['image'], sample['dx']) for sample in chunk]
            test_samples.extend(chunk_samples)
    else:
        total_size = len(dataset)
        indices = range(total_size)

        labels = [sample['dx'] for sample in dataset]
        _, sampled_indices = train_test_split(
            indices,
            test_size=size / total_size,
            stratify=labels,
            random_state=42
        )

        test_samples = []
        for dx_index in tqdm(sampled_indices, desc='Processing samples'):
            sample = dataset[dx_index]
            test_samples.append((sample['image'], sample['dx']))

    return test_samples
