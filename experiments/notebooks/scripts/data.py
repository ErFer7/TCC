'''
Preparação dos dados de treinamento.
'''

from os.path import join
from json import dump
from re import search

from pydantic import BaseModel

import scripts.definitions as defs

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
    location: str
    distribution: list[str]
    risk: str
    skin_lesion: str
    conclusion: str


class RawLesionData(BaseModel):
    '''
    Modelo de exame de aproximação.
    '''

    exam_id: int
    images: list[str]
    lesion_number: int
    lesion_location: str
    report: str | Report


class LesionData(BaseModel):
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
    location_distribution: Distribution
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


def parse_list_report_part(structured_report: Report,
                           report_parts: list[str],
                           attribute_name: str,
                           domain: tuple[str],
                           optional: bool = False) -> None:
    '''
    Processa uma parte do laudo.
    '''

    while True:
        if report_parts[0] in domain:  # type: ignore
            getattr(structured_report, attribute_name).append(report_parts.pop(0))
        elif not optional and len(getattr(structured_report, attribute_name)) == 0:
            raise ValueError(f'Instrutura de laudo incorreta: {report_parts[0]}')
        else:
            break


def parse_string_report_part(structured_report: Report,
                             report_parts: list[str],
                             attribute_name: str,
                             domain: tuple[str] | None) -> None:
    '''
    Processa uma parte do laudo.
    '''

    if domain is None:
        setattr(structured_report, attribute_name, report_parts.pop(0))
        return

    if report_parts[0] in domain:  # type: ignore
        setattr(structured_report, attribute_name, report_parts.pop(0))
    else:
        raise ValueError(f'Instrutura de laudo incorreta: {report_parts[0]}')


def next_lines_contain_report(report_parts: list[str]) -> bool:
    '''
    Verifica se as próximas linhas contêm laudo.
    '''

    if len(report_parts) < 8:
        return False

    if search(r'Lesão \d+.', report_parts[0]):
        return False

    return report_parts[0] in defs.ELEMENTARY_LESIONS_DOMAIN


def parse_report(report_parts: list[str]) -> tuple[Report | None, bool]:
    '''
    Processa do laudo.
    Retorna um laudo estruturado e um booleano indicando se há mais laudos.
    '''

    if search(r'Lesão \d+.', report_parts[0]):
        return None, False

    structured_report = Report(
        elementary_lesions=[],
        secondary_lesions=[],
        coloration=[],
        morphology=[],
        size='',
        location='',
        distribution=[],
        risk='',
        skin_lesion='',
        conclusion='',
    )

    parse_list_report_part(
        structured_report,
        report_parts,
        'elementary_lesions',
        defs.ELEMENTARY_LESIONS_DOMAIN  # type: ignore
    )

    parse_list_report_part(
        structured_report,
        report_parts,
        'secondary_lesions',
        defs.SECONDARY_LESIONS_DOMAIN,  # type: ignore
        True
    )

    parse_list_report_part(
        structured_report,
        report_parts,
        'coloration',
        defs.COLORATION_DOMAIN  # type: ignore
    )

    parse_list_report_part(
        structured_report,
        report_parts,
        'morphology',
        defs.MORPHOLOGY_DOMAIN  # type: ignore
    )

    parse_string_report_part(
        structured_report,
        report_parts,
        'size',
        defs.SIZE_DOMAIN  # type: ignore
    )

    parse_string_report_part(
        structured_report,
        report_parts,
        'location',
        None  # type: ignore
    )

    parse_list_report_part(
        structured_report,
        report_parts,
        'distribution',
        defs.DISTRIBUTION_DOMAIN  # type: ignore
    )

    parse_string_report_part(
        structured_report,
        report_parts,
        'risk',
        defs.RISK_DOMAIN  # type: ignore
    )

    parse_string_report_part(
        structured_report,
        report_parts,
        'skin_lesion',
        defs.SKIN_LESION_DOMAIN  # type: ignore
    )

    conclusion = ''

    while len(report_parts) > 0 and \
            not next_lines_contain_report(report_parts) and \
            not search(r'Lesão \d+.', report_parts[0]):
        conclusion += report_parts.pop(0) + '\n'

    structured_report.conclusion = conclusion

    if len(structured_report.secondary_lesions) == 0:
        structured_report.secondary_lesions = ['Nenhuma']

    structured_report.size = defs.SIZE_DOMAIN_TRANSFORMED[structured_report.size]

    if structured_report.risk == 'AMARELA - ENCAMINHAMENTO COM PRIORIDADE PARA O AMBULATÓRIO DE REFERÊNCIA':
        structured_report.risk = 'AMARELA - ENCAMINHAMENTO COM PRIORIDADE PARA O AMBULATÓRIO DE REFERÊNCIA TERCIÁRIO'

    return structured_report, True


def analyse_dataset(dataset: list[LesionData],
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
        location_distribution=Distribution(classes_count=0, classes={}),
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
        'location',
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
