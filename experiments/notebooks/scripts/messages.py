'''
Mensagens.
'''

from os.path import join

from scripts.data import DatasetAnalysis, LesionData

import scripts.definitions as defs


def format_prompt(prompt_type: defs.PromptType, dataset_analysis: DatasetAnalysis) -> str:
    '''
    Formata o prompt.
    '''

    match prompt_type:
        case defs.PromptType.SIMPLE_CLASSIFICATION:
            prompt_template = defs.SIMPLE_CLASSIFICATION_PROMPT_TEMPLATE

            return prompt_template.format(', '.join(dataset_analysis.skin_lesion_distribution.classes.keys()))
        case defs.PromptType.REPORT:
            prompt_template = defs.REPORT_PROMPT_TEMPLATE

            return prompt_template.format(
                ', '.join(dataset_analysis.elementary_lesions_distribution.classes.keys()),
                ', '.join(dataset_analysis.secondary_lesions_distribution.classes.keys()),
                ', '.join(dataset_analysis.coloration_distribution.classes.keys()),
                ', '.join(dataset_analysis.morphology_distribution.classes.keys()),
                ', '.join(dataset_analysis.size_distribution.classes.keys()),
                ', '.join(dataset_analysis.skin_lesion_distribution.classes.keys()),
                ', '.join(dataset_analysis.risk_distribution.classes.keys())
            )
        case _:
            raise ValueError(f'Tipo de prompt inválido: {prompt_type}')


def format_answer(prompt_type: defs.PromptType, lesion_data: LesionData) -> str:
    '''
    Formata a resposta.
    '''

    match prompt_type:
        case defs.PromptType.SIMPLE_CLASSIFICATION:
            answer_template = defs.SIMPLE_CLASSIFICATION_ANSWER_TEMPLATE

            return answer_template.format(lesion_data.report.skin_lesion)
        case defs.PromptType.REPORT:
            answer_template = defs.REPORT_ANSWER_TEMPLATE

            return answer_template.format(
                ', '.join(lesion_data.report.elementary_lesions),
                ', '.join(lesion_data.report.secondary_lesions),
                ', '.join(lesion_data.report.coloration),
                ', '.join(lesion_data.report.morphology),
                lesion_data.report.size,
                lesion_data.report.skin_lesion,
                lesion_data.report.risk,
                lesion_data.report.conclusion
            )
        case _:
            raise ValueError(f'Tipo de prompt inválido: {prompt_type}')


def add_inference_message(prompt: str, messages: list | None = None) -> list:
    '''
    Adiciona uma mensagem na lista de mensagens.
    '''

    messages = messages if messages is not None else []
    messages.append(
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image'
                },
                {
                    'type': 'text',
                    'text': prompt
                }
            ]
        }
    )

    return messages


def create_training_message(prompt_type: defs.PromptType,
                            lesion_data: LesionData,
                            dataset_analysis: DatasetAnalysis) -> dict:
    '''
    Formata os dados.
    '''

    image_path = join(defs.DATA_PATH, 'images', lesion_data.image)

    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': format_prompt(prompt_type, dataset_analysis),
                    }, {
                        'type': 'image',
                        'image': f'file://{image_path}',
                    }
                ],
            },
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': format_answer(prompt_type, lesion_data)}],
            },
        ],
    }
