'''
Preparação dos dados de treinamento.
'''

from os.path import join
from json import dump, load
from tqdm.notebook import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split

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


def analyse_dataset(data: list, dataset_path: str, dataset_name: str, save: bool = True) -> dict:
    '''
    Analisa os dados.
    '''

    data_analysis = {
        'total_size': 0,
        'average_images_per_exam': 0,
        'elementary_lesions_distribution': {'classes_count': 0, 'classes': {}},
        'secondary_lesions_distribution': {'classes_count': 0, 'classes': {}},
        'coloration_distribution': {'classes_count': 0, 'classes': {}},
        'morphology_distribution': {'classes_count': 0, 'classes': {}},
        'size_distribution': {'classes_count': 0, 'classes': {}},
        'local_distribution': {'classes_count': 0, 'classes': {}},
        'distribution_distribution': {'classes_count': 0, 'classes': {}},
        'risk_distribution': {'classes_count': 0, 'classes': {}},
        'skin_lesion_distribution': {'classes_count': 0, 'classes': {}}
    }


    data_analysis['total_size'] = len(data)
    data_analysis['average_images_per_exam'] = sum(len(exam['images'])
                                                   for exam in data) / data_analysis['total_size']

    lists = [
        'elementary_lesions',
        'secondary_lesions',
        'coloration',
        'morphology',
        'distribution'
    ]

    strings = [
        'size',
        'local',
        'risk',
        'skin_lesion'
    ]

    for exam in data:
        report = exam['report']

        for data_list in lists:
            for value in report[data_list]:
                distribution = data_analysis[f'{data_list}_distribution']

                if value not in distribution['classes']:
                    distribution['classes_count'] += 1
                    distribution['classes'][value] = 0

                distribution['classes'][value] += 1

        for data_string in strings:
            distribution = data_analysis[f'{data_string}_distribution']

            if report[data_string] not in distribution['classes']:
                distribution['classes_count'] += 1
                distribution['classes'][report[data_string]] = 0

            distribution['classes'][report[data_string]] += 1

    for key, distribution in data_analysis.items():
        if key.endswith('_distribution'):
            classes = distribution['classes']

            for value in classes:
                classes[value] = {'value': classes[value],
                                  'proportion': round(classes[value] / data_analysis['total_size'] * 100, 3)}

            sorted_items = sorted(distribution['classes'].items(), key=lambda x: x[1]['value'], reverse=True)
            data_analysis[key]['classes'] = dict(sorted_items)

    if save:
        with open(join(dataset_path, f'{dataset_name[:-5]}_analysis.json'), 'w', encoding='utf-8') as file:
            dump(data_analysis, file, indent=4, ensure_ascii=False)

    return data_analysis
