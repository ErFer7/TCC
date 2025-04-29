'''
Módulo para utilitários de treinamento.
'''

from typing import Any

from pydantic import BaseModel

from scripts.definitions import PromptType


class Training(BaseModel):
    '''
    Hiperparâmetros de treinamento.
    '''

    base_model_name: str
    trained_model_name: str
    quantization: bool
    prompt_type: PromptType
    version: str
    size: int
    peft_hyperparameters: dict[str, Any]
    sft_hyperparameters: dict[str, Any]
    used_memory: float
    training_time: float
