'''
Preparação dos dados de treinamento.
'''

from os.path import join
from json import dump
from pydantic import BaseModel

PROMPT = 'Classify the skin lesion in the image.'
ANSWER = 'The skin lesion in the image is {disease}.'


class Series(BaseModel):
    '''
    Modelo de série.
    '''

    seriesdescription: str
    seriesinstanceuid: str
    seriesnumber: int
    instances: list[str]


class RawData(BaseModel):
    '''
    Modelo de dados brutos.
    '''

    id_exame: int
    series: list[Series]


class RawApproximationExam(BaseModel):
    '''
    Modelo de exame de aproximação.
    '''

    exam_id: int
    images: list[str]
    report: str


class RawReport(BaseModel):
    '''
    Modelo de laudo bruto.
    '''

    id_solicitacao: int
    id_exame: int
    id_laudo: int
    laudo: str


class Report(BaseModel):
    '''
    Modelo de laudo.
    '''

    elementary_lesions: list[str]
    secondary_lesions: list[str]
    coloration: list[str]
    morphology: list[str]
    size: str
    local: str
    distribution: list[str]
    risk: str
    skin_lesion: str
    conclusion: str


class ApproximationExam(BaseModel):
    '''
    Modelo de exame de aproximação estruturado.
    '''

    exam_id: int
    image: str
    report: Report


class ClassCount(BaseModel):
    '''
    Modelo de contagem de classes.
    '''

    count: int
    proportion: float


class Distribution(BaseModel):
    '''
    Modelo de distribuição.
    '''

    classes_count: int
    classes: dict[str, ClassCount]


class DatasetAnalysis(BaseModel):
    '''
    Modelo de análise de dados.
    '''

    total_size: int
    elementary_lesions_distribution: Distribution
    secondary_lesions_distribution: Distribution
    coloration_distribution: Distribution
    morphology_distribution: Distribution
    size_distribution: Distribution
    local_distribution: Distribution
    distribution_distribution: Distribution
    risk_distribution: Distribution
    skin_lesion_distribution: Distribution


# TODO: Mover para messages.py
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


def analyse_dataset(dataset: list[ApproximationExam],
                    dataset_path: str,
                    dataset_name: str,
                    save: bool = True) -> DatasetAnalysis:
    '''
    Analisa os dados.
    '''

    data_analysis = DatasetAnalysis(
        total_size=len(dataset),
        elementary_lesions_distribution=Distribution(classes_count=0, classes={}),
        secondary_lesions_distribution=Distribution(classes_count=0, classes={}),
        coloration_distribution=Distribution(classes_count=0, classes={}),
        morphology_distribution=Distribution(classes_count=0, classes={}),
        size_distribution=Distribution(classes_count=0, classes={}),
        local_distribution=Distribution(classes_count=0, classes={}),
        distribution_distribution=Distribution(classes_count=0, classes={}),
        risk_distribution=Distribution(classes_count=0, classes={}),
        skin_lesion_distribution=Distribution(classes_count=0, classes={}),
    )

    lists = (
        'elementary_lesions',
        'secondary_lesions',
        'coloration',
        'morphology',
        'distribution'
    )

    strings = (
        'size',
        'local',
        'risk',
        'skin_lesion'
    )

    for data in dataset:
        report = data.report

        for data_list in lists:
            for value in getattr(report, data_list):
                distribution: Distribution = getattr(data_analysis, f'{data_list}_distribution')

                if value not in distribution.classes:
                    distribution.classes_count += 1
                    distribution.classes[value] = ClassCount(count=0, proportion=0.0)

                distribution.classes[value].count += 1

        for data_string in strings:
            distribution: Distribution = getattr(data_analysis, f'{data_string}_distribution')
            report_value: str = getattr(report, data_string)

            if report_value not in distribution.classes:
                distribution.classes_count += 1
                distribution.classes[report_value] = ClassCount(count=0, proportion=0.0)

            distribution.classes[report_value].count += 1

    for key, distribution in data_analysis.model_dump().items():
        if key.endswith('_distribution'):
            classes = distribution['classes']  # type: ignore

            for value in classes:
                classes[value]['proportion'] = round(classes[value]['count'] / data_analysis.total_size * 100, 3)

            sorted_classes = sorted(classes.items(), key=lambda class_: class_[1]['count'], reverse=True)
            distribution['classes'] = dict(sorted_classes)  # type: ignore
            new_distribuition = Distribution(**distribution)  # type: ignore

            setattr(data_analysis, key, new_distribuition)

    if save:
        with open(join(dataset_path, f'{dataset_name[:-5]}_analysis.json'), 'w', encoding='utf-8') as file:
            dump(data_analysis.model_dump(), file, indent=4, ensure_ascii=False)

    return data_analysis
