'''
Autenticação.
'''

from os import getenv
from os.path import exists
from getpass import getpass
from dotenv import load_dotenv
from huggingface_hub import login


def authenticate_huggingface() -> None:
    '''
    Autenticação no Hugging Face.
    '''

    hf_token = None

    if not exists('../.env'):
        hf_token = getpass('Enter your Hugging Face token: ')
    else:
        load_dotenv(dotenv_path='../.env')
        hf_token = getenv('HF_TOKEN')

    assert hf_token is not None, 'Invalid Hugging Face token'

    login(token=hf_token)
