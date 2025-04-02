'''
Constantes.
'''

from enum import Enum
from os.path import join

DATA_PATH = join('..', 'data')
ANALYSIS_PATH = join('..', 'analysis')
RESULTS_PATH = join('..', 'results')
TRAINING_PATH = join('..', 'training')

TRAINING_PROPORTION = 0.8
VALIDATION_PROPORTION = 0.1
TEST_PROPORTION = 0.1

BASE_MODEL_NAME = 'weights/LlaDerm-R-0.1-11B-4bit'
BASE_MODEL_NAME = 'unsloth/Llama-3.2-11B-Vision-Instruct'

STATIC_RANDOM_STATE = 3407

# TODO: Melhorar os prompts
SIMPLE_CLASSIFICATION_PROMPT_TEMPLATE = 'Classifique a lesão de pele na imagem. Não inclua nenhum comentário extra ' \
    'na resposta além da classificação.\nAs opções de classificação são: {}.'
REPORT_PROMPT_TEMPLATE = 'Classifique a lesão de pele na imagem, informando a lesão elementar, lesão ' \
    'secundária, coloração, morfologia, tamanho em centímetros, classificação da ' \
    'lesão e classificação de risco. Por fim, inclua uma breve conclusão sobre o diganóstico.' \
    '\nAs opções de lesões elementares são: {}.\nAs opções de lesões secundárias são: {}.\nAs ' \
    'opções de coloração são: {}.\nAs opções de morfologia são: {}.\nAs opções de ' \
    'tamanho são: {}.\nAs opções de classificação são: {}.\nAs opções de classificação de risco são: {}.'
SIMPLE_CLASSIFICATION_ANSWER_TEMPLATE = '{}.'
REPORT_ANSWER_TEMPLATE = 'Lesão elementar: {}.\n\nLesão secundária: {}.\n\nColoração: {}.\n\n' \
                         'Morfologia: {}.\n\nTamanho: {}.\n\nClassificação: {}.\n\n' \
                         'Classificação de risco: {}.\n\nConclusão: {}.'

ELEMENTARY_LESIONS_DOMAIN = (
    'Mácula/mancha',
    'Pápula',
    'Placa',
    'Nódulo',
    'Vesícula',
    'Pústula',
    'Bolha',
    'Cisto',
    'Comedão',
    'Urtica/ponfo',
    'Púrpura',
    'Petéquia',
    'Equimose',
    'Telangectasias',
    'Úlcera',
    'Ausente',
    'Tumor'
)

SECONDARY_LESIONS_DOMAIN = (
    'Escamas',
    'Crostas',
    'Exulceração',
    'Erosão',
    'Fissura',
    'Liquenificação',
    'Atrofia',
    'Cicatriz',
    'Ausente',
    'Escoriação',
    'Ceratose',
    'Alopécia',
    'Maceração'
    'Nenhuma'  # Adicionada para o treinamento
)

COLORATION_DOMAIN = (
    'Eritematosa (avermelhada)',
    'Castanha',
    'Negra',
    'Perlácea',
    'Violácea',
    'Azulada',
    'Hipo/acrômica (despigmentada)',
    'Eucrômica',
    'Amarelada',
    'Eucrômica'
)

MORPHOLOGY_DOMAIN = (
    'Linear',
    'Zosteriforme',
    'Gutata',
    'Lenticular',
    'Anular',
    'Numular',
    'Policíclica',
    'Circinada',
    'Circular ou Arredondada',
    'Irregular/assimétrica',
    'Séssil / Pedunculada',
    'Papilomatosa / Verrucosa',
    'Intertriginosa',
    'Arboriforme',
    'Puntiforme',
    'Folicular'
)

SIZE_DOMAIN = (
    '< 1',
    '1 a 2',
    '2 a 4',
    '> 4'
)

DISTRIBUTION_DOMAIN = (
    'Única',
    'Localizada',
    'Disseminada',
    'Generalizada'
)

SKIN_LESION_DOMAIN = (
    'Lesões de pele que necessitam de avaliação presencial com dermatologista',
    'Ceratose seborreica',
    'Câncer de pele Não Melanoma (CEC ou CBC)',
    'Ceratoses solares/actínicas',
    'Nevo melanocítico',
    'Outras',
    'Fotodano crônico',
    'Lesões de pele que necessitam de avaliação presencial com dermatologista pediátrico',
    'Dermatite de Contato',
    'Câncer de pele Melanoma',
    'Psoríase moderada a grave',
    'Anexos/alopécias',
    'Lesões de pele benignas com indicação de tratamento cirúrgico',
    'Verrugas Virais',
    'Outras dermatoses',
    'Dermatite Atópica',
    'Pitiríase Versicolor',
    'Psoríase leve',
    'Acne',
    'Tineas',
    'Foliculite',
    'Outras Especialidades',
    'Melasma',
    'Líquen simples crônico (Neurodermite)',
    'Asteatose Cutânea',
    'Eritematosa (avermelhada)',
    'Eczema de Estase',
    'Dermatite Seborrêica',
    'Molusco Contagioso',
    'Eczemas',
    'Rosácea',
    'Anexos/onicoses',
    'Escabiose',
    'Estrófulo',
    'Fototerapia',
    'Prurido',
    'Colagenoses',
    'Bolhosas',
    'Onicomicose',
    'Urticária',
    'Impetigo',
    'Candidíase - Intertrigo',
    'Castanha',
    'Hanseníase',
    'Alopécias Adquiridas',
    'Cirurgia Vascular',
    'Pitiríase Rósea',
    'Herpes Simples Recidivante',
    'Furunculose',
    'Reumatologista',
    'Oftalmo-cirurgia/Oftalmo-plástica'
)

RISK_DOMAIN = (
    'VERMELHA - QUADROS AGUDOS E GRAVES',
    'AMARELA - ENCAMINHAMENTO COM PRIORIDADE PARA O AMBULATÓRIO DE REFERÊNCIA TERCIÁRIO',
    'AMARELA - ENCAMINHAMENTO COM PRIORIDADE PARA O AMBULATÓRIO DE REFERÊNCIA',  # Versão alternativa
    'VERDE - AVALIAÇÃO CLÍNICO-CIRURGIA COM ESPECIALISTA',
    'AZUL - TRATAMENTO NA UNIDADE BÁSICA DE SAÚDE (UBS)',
    'BRANCA - SEM NECESSIDADE DE INTERVENÇÃO OU ACOMPANHAMENTO'
)


class PromptType(Enum):
    '''
    Tipos de prompt.
    '''

    SIMPLE_CLASSIFICATION = 'simple_classification'
    REPORT = 'report'
