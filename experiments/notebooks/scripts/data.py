'''
Preparação dos dados de treinamento.
'''

from os.path import join
from json import dump
from re import search

from pydantic import BaseModel

import scripts.definitions as defs


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


def check_string_report_part(report_parts: list[str],
                             domain: tuple[str] | None,
                             index: int = 0) -> tuple[int, bool]:
    '''
    Verifica se uma parte do laudo está presente.
    Retorna o índice da parte do laudo e um booleano indicando se a parte do laudo está presente.
    '''

    has_part = False

    if domain is None:
        index += 1
        has_part = True
        return index, has_part

    if report_parts[index] in domain:  # type: ignore
        index += 1
        has_part = True

    return index, has_part


def next_lines_contain_report(report_parts: list[str]) -> bool:
    '''
    Verifica se as próximas linhas contêm laudo.
    '''

    if len(report_parts) < 8:
        return False

    index = 0
    has_elementary_lesion = False

    while True:
        if report_parts[index] in defs.ELEMENTARY_LESIONS_DOMAIN and \
           not (has_elementary_lesion and report_parts[index] == 'Ausente'):
            index += 1
            has_elementary_lesion = True
        elif not has_elementary_lesion:
            return False
        else:
            break

    while True:
        if report_parts[index] in defs.SECONDARY_LESIONS_DOMAIN:
            index += 1
        else:
            break

    has_coloration = False

    while True:
        if report_parts[index] in defs.COLORATION_DOMAIN:
            index += 1
            has_coloration = True
        elif not has_coloration:
            return False
        else:
            break

    has_morphology = False

    while True:
        if report_parts[index] in defs.MORPHOLOGY_DOMAIN:
            index += 1
            has_morphology = True
        elif not has_morphology:
            return False
        else:
            break

    if report_parts[index] in defs.SIZE_DOMAIN:
        index += 1
    else:
        return False

    index += 1

    has_distribution = False

    while True:
        if report_parts[index] in defs.DISTRIBUTION_DOMAIN:
            index += 1
            has_distribution = True
        elif not has_distribution:
            return False
        else:
            break

    if report_parts[index] in defs.RISK_DOMAIN:
        index += 1
    else:
        return False

    if report_parts[index] in defs.SKIN_LESION_DOMAIN:
        index += 1
    else:
        return False

    return True


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

    while True:
        if report_parts[0] in defs.ELEMENTARY_LESIONS_DOMAIN and \
           not (len(structured_report.elementary_lesions) > 0 and report_parts[0] == 'Ausente'):
            structured_report.elementary_lesions.append(report_parts.pop(0))
        elif len(structured_report.elementary_lesions) == 0:
            raise ValueError(f'Instrutura de laudo incorreta: {report_parts[0]}')
        else:
            break

    while True:
        if report_parts[0] in defs.SECONDARY_LESIONS_DOMAIN:
            structured_report.secondary_lesions.append(report_parts.pop(0))
        else:
            break

    while True:
        if report_parts[0] in defs.COLORATION_DOMAIN:
            structured_report.coloration.append(report_parts.pop(0))
        elif len(structured_report.coloration) == 0:
            raise ValueError(f'Instrutura de laudo incorreta: {report_parts[0]}')
        else:
            break

    while True:
        if report_parts[0] in defs.MORPHOLOGY_DOMAIN:
            structured_report.morphology.append(report_parts.pop(0))
        elif len(structured_report.morphology) == 0:
            raise ValueError(f'Instrutura de laudo incorreta: {report_parts[0]}')
        else:
            break

    if report_parts[0] in defs.SIZE_DOMAIN:
        structured_report.size = report_parts.pop(0)
    else:
        raise ValueError(f'Instrutura de laudo incorreta: {report_parts[0]}')

    structured_report.location = report_parts.pop(0)

    while True:
        if report_parts[0] in defs.DISTRIBUTION_DOMAIN:
            structured_report.distribution.append(report_parts.pop(0))
        elif len(structured_report.distribution) == 0:
            raise ValueError(f'Instrutura de laudo incorreta: {report_parts[0]}')
        else:
            break

    if report_parts[0] in defs.RISK_DOMAIN:
        structured_report.risk = report_parts.pop(0)
    else:
        raise ValueError(f'Instrutura de laudo incorreta: {report_parts[0]}')

    if report_parts[0] in defs.SKIN_LESION_DOMAIN:
        structured_report.skin_lesion = report_parts.pop(0)
    else:
        raise ValueError(f'Instrutura de laudo incorreta: {report_parts[0]}')

    conclusion = ''

    while len(report_parts) > 0 and \
            not next_lines_contain_report(report_parts) and \
            not search(r'Lesão \d+.', report_parts[0]):
        conclusion += report_parts.pop(0) + '\n'

    structured_report.conclusion = conclusion

    if len(structured_report.secondary_lesions) == 0:
        structured_report.secondary_lesions = ['Ausente']

    structured_report.size = defs.SIZE_DOMAIN_TRANSFORMED[structured_report.size]

    if structured_report.risk == 'AMARELA - ENCAMINHAMENTO COM PRIORIDADE PARA O AMBULATÓRIO DE REFERÊNCIA':
        structured_report.risk = 'AMARELA - ENCAMINHAMENTO COM PRIORIDADE PARA O AMBULATÓRIO DE REFERÊNCIA TERCIÁRIO'

    return structured_report, True


def parse_report_footnote(report_parts: list[str]) -> dict[int, list[str]]:
    '''
    Processa a nota de rodapé do laudo.
    '''

    lesions = {}
    last_lesion = 0

    while len(report_parts) > 0:
        if search(r'Lesão \d+.', report_parts[0]):
            lesion_conclusion = report_parts.pop(0)
            conclusion_parts = lesion_conclusion.split()
            lesion_number = int(conclusion_parts[1].strip(':'))
            last_lesion = lesion_number

            if lesion_number not in lesions:
                lesions[lesion_number] = []

            lesions[lesion_number].append(' '.join(conclusion_parts[2:]))
        else:
            lesions[last_lesion].append(report_parts.pop(0))

    return lesions


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
