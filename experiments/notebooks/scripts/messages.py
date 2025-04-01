'''
Mensagens.
'''

from scripts.configuration import PromptType


def format_prompt(prompt_template: str, prompt_type: PromptType, dataset_analysis: dict) -> str | None:
    '''
    Formata o prompt.
    '''

    match prompt_type:
        case PromptType.SIMPLE_CLASSIFICATION:
            return prompt_template.format(dataset_analysis['skin_lesion_distribution']['classes'].keys())
        case PromptType.FULL_CLASSIFICATION:
            return prompt_template.format(
                dataset_analysis['elementary_lesions_distribution']['classes'].keys(),
                dataset_analysis['secondary_lesions_distribution']['classes'].keys(),
                dataset_analysis['coloration_distribution']['classes'].keys(),
                dataset_analysis['morphology_distribution']['classes'].keys(),
                dataset_analysis['size_distribution']['classes'].keys(),
                dataset_analysis['skin_lesion_distribution']['classes'].keys()
            )
        case PromptType.REPORT:
            return prompt_template.format(
                dataset_analysis['elementary_lesions_distribution']['classes'].keys(),
                dataset_analysis['secondary_lesions_distribution']['classes'].keys(),
                dataset_analysis['coloration_distribution']['classes'].keys(),
                dataset_analysis['morphology_distribution']['classes'].keys(),
                dataset_analysis['size_distribution']['classes'].keys(),
                dataset_analysis['skin_lesion_distribution']['classes'].keys(),
                dataset_analysis['risk_distribution']['classes'].keys()
            )
        case _:
            raise ValueError('Invalid prompt type.')

    return None


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
