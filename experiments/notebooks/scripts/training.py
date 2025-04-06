'''
Módulo para utilitários de treinamento.
'''

from typing import Any

from pydantic import BaseModel

from scripts.definitions import PromptType


class TrainingHyperparameters(BaseModel):
    '''
    Hiperparâmetros de treinamento.
    '''

    # Parâmetros básicos de identificação e treinamento
    model_name: str
    quantization: bool
    prompt_type: PromptType
    version: str
    size: int

    # Hiperparâmetros de PEFT
    finetune_vision_layers: bool
    finetune_language_layers: bool
    finetune_attention_modules: bool
    finetune_mlp_modules: bool
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    use_rslora: bool
    loftq_config: Any

    # Hiperparâmetros de treinamento
    eval_strategy: str
    learning_rate: float
    weight_decay: float
    lr_scheduler_type: str
    optim: str
