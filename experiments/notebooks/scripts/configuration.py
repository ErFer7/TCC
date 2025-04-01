'''
Constantes.
'''

from enum import Enum
from os.path import join

DATA_PATH = join('..', 'data')
ANALYSIS_PATH = join('..', 'analysis')
RESULTS_PATH = join('..', 'results')

TRAINING_PROPORTION = 0.8
VALIDATION_PROPORTION = 0.1
TEST_PROPORTION = 0.1

BASE_MODEL_NAME = 'weights/LlaDerm-R-0.1-11B-4bit'
BASE_MODEL_NAME = 'unsloth/Llama-3.2-11B-Vision-Instruct'

# TODO: Melhorar os prompts
SIMPLE_CLASSIFICATION_PROMPT_TEMPLATE = 'Classifique a lesão de pele na imagem. Não inclua nenhum comentário extra ' \
    'na resposta além da classificação.\nAs opções de classificação são: {}.'
# TODO: Avaliar a necessidade
FULL_CLASSIFICATION_PROMPT_TEMPLATE = 'Classifique a lesão de pele na imagem, informando a lesão primária, lesão ' \
    'secundária, coloração, morfologia, tamanho em centímetros e a classificação da ' \
    'lesão. Não inclua nenhum comentário além das partes requisitadas.\nAs opções ' \
    'de lesões primárias são: {}.\nAs opções de lesões secundárias são: {}.\nAs ' \
    'opções de coloração são: {}.\nAs opções de morfologia são: {}.\nAs opções de ' \
    'tamanho são: {}.\nAs opções de classificação são: {}.'
REPORT_PROMPT_TEMPLATE = 'Classifique a lesão de pele na imagem, informando a lesão primária, lesão ' \
    'secundária, coloração, morfologia, tamanho em centímetros, classificação da ' \
    'lesão e classificação de risco. Por fim, inclua uma breve conclusão sobre o diganóstico.' \
    '\nAs opções de lesões primárias são: {}.\nAs opções de lesões secundárias são: {}.\nAs ' \
    'opções de coloração são: {}.\nAs opções de morfologia são: {}.\nAs opções de ' \
    'tamanho são: {}.\nAs opções de classificação são: {}.\nAs opções de classificação de risco são: {}.'
SIMPLE_CLASSIFICATION_ANSWER_TEMPLATE = '{}.'
FULL_CLASSIFICATION_ANSWER_TEMPLATE = ''
REPORT_ANSWER_TEMPLATE = ''


class PromptType(Enum):
    '''
    Tipos de prompt.
    '''

    SIMPLE_CLASSIFICATION = 'simple_classification'
    FULL_CLASSIFICATION = 'full_classification'
    REPORT = 'report'
