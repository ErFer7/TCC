'''
Módulo de utilitários para testes.
'''

from datetime import datetime

from pydantic import BaseModel

from scripts.definitions import Model


class GenerationParameters(BaseModel):
    '''
    Parâmetros de geração.
    '''

    max_new_tokens: int
    use_cache: bool
    do_sample: bool
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

    model_name: str
    model: Model
    generation_parameters: GenerationParameters
    start_time: datetime
    end_time: datetime | None
    results_on_test_data: list[TestResult]
    results_on_training_data: list[TestResult]
