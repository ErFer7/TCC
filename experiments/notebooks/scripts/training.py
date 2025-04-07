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

    base_model_name: str
    model_name: str
    quantization: bool
    prompt_type: PromptType
    version: str
    size: int
    peft_hyperparameters: dict[str, Any]
    sft_hyperparameters: dict[str, Any]

# TODO: Ideias...
# def custom_loss(logits, labels):
#     loss = F.cross_entropy(logits, labels, reduction="none")

#     # Identify positions of structured fields (e.g., first 6 lines)
#     structured_positions = (labels < CONCLUSION_START_TOKEN_ID)  # Adjust based on tokenization
#     loss[structured_positions] *= 3.0  # Amplify loss for structured fields
#     return loss.mean()
