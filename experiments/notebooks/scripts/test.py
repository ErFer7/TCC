'''
Módulo de utilitários para testes.
'''

from pydantic import BaseModel

from scripts.definitions import Model


class GenerationParameters(BaseModel):
    '''
    Parâmetros de geração.
    '''

    max_new_tokens: int
    temperature: float


class TestResult(BaseModel):
    '''
    Resultado do teste.
    '''

    exam_id: int
    image: str
    answer: str


class Test(BaseModel):
    '''
    Teste.
    '''

    tested_model: str
    model: Model
    generation_parameters: GenerationParameters
    results_on_test_data: list[TestResult]
    results_on_training_data: list[TestResult]
