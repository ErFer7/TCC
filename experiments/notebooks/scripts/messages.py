'''
Mensagens.
'''


def add_inference_message(message: str, messages: list | None = None) -> list:
    '''
    Mensagem de adição de inferência.
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
                    'text': message
                }
            ]
        }
    )

    return messages
