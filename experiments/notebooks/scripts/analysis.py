'''
Análise de resultados.
'''

# TODO: Colocar tudo nos módulos corretos

from unicodedata import normalize, combining

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from seaborn import heatmap
from pydantic import BaseModel
from tensorboard.backend.event_processing import event_accumulator

import matplotlib.pyplot as plt
import pandas as pd

from scripts.test import TestResult
from scripts.data import SimpleLesionData, ClassCount
from scripts.messages import format_answer

import scripts.definitions as defs


class Answer(BaseModel):
    '''
    Resposta.
    '''

    valid: bool


class SimpleClassificationAnswer(Answer):
    '''
    Resposta da classificação simples.
    '''

    skin_lesion: str


class ReportClassification(BaseModel):
    '''
    Classificação do laudo. Inclui apenas os campos relevantes para o estudo e de classificação.
    '''

    skin_lesion: str
    risk: str


class ReportClassificationAnswer(Answer, ReportClassification):
    '''
    Resposta do laudo.
    '''


class InvalidAnswer(Answer):
    '''
    Resposta inválida.
    '''

    answer: str


class AnswerResult(BaseModel):
    '''
    Resultado.
    '''

    exam_id: int
    image: str
    answer: SimpleClassificationAnswer | ReportClassificationAnswer | InvalidAnswer


class SimpleResultPair(BaseModel):
    '''
    Par de resultados.
    '''

    exam_id: int
    image: str
    answer: str
    expected: str


class SimpleClassificationResultPair(SimpleResultPair):
    '''
    Par de resultados da classificação simples.
    '''


class ReportClassificationResultPair(BaseModel):
    '''
    Par de resultados do laudo.
    '''

    exam_id: int
    image: str
    answer: ReportClassification
    expected: ReportClassification


class TestAnalysis(BaseModel):
    '''
    Análise de teste.
    '''

    test_name: str
    valid_results_on_test_data: list[AnswerResult]
    valid_results_on_training_data: list[AnswerResult]
    invalid_results_on_test_data: list[AnswerResult]
    invalid_results_on_training_data: list[AnswerResult]


def sanitize_domain_class(domain_class: str) -> str:
    '''
    Sanitiza a classe do domínio.
    '''

    domain_class = domain_class.strip().strip('.,*')
    domain_class = domain_class.lower()
    domain_class = normalize('NFKD', domain_class)
    domain_class = ''.join([c for c in domain_class if not combining(c)])
    domain_class = domain_class.replace(' ', '_')

    return domain_class


elementary_lesion_classes = tuple(map(sanitize_domain_class, defs.ELEMENTARY_LESIONS_DOMAIN))
secondary_lesion_classes = tuple(map(sanitize_domain_class, defs.SECONDARY_LESIONS_DOMAIN))
coloration_classes = tuple(map(sanitize_domain_class, defs.COLORATION_DOMAIN))
morphology_classes = tuple(map(sanitize_domain_class, defs.MORPHOLOGY_DOMAIN))
size_classes = tuple(map(sanitize_domain_class, defs.SIZE_DOMAIN))
skin_lesion_classes = tuple(map(sanitize_domain_class, defs.SKIN_LESION_DOMAIN))
risk_classes = tuple(map(sanitize_domain_class, defs.RISK_DOMAIN))


def structure_simple_classification_answer(answer: str) -> SimpleClassificationAnswer:
    '''
    Estrutura a resposta.
    '''

    answer = answer.strip().strip('.')
    sanitized_answer = sanitize_domain_class(answer)

    valid = sanitized_answer in skin_lesion_classes

    return SimpleClassificationAnswer(
        valid=valid,
        skin_lesion=answer
    )


def structure_report_classification_answer(answer: str) -> ReportClassificationAnswer | InvalidAnswer:
    '''
    Estrutura a resposta.
    '''

    answer_parts = answer.splitlines()

    skin_lesion = answer_parts[0].split(':', 1)[1].strip().strip('.')
    risk = answer_parts[1].split(':', 1)[1].strip().strip('.')

    sanitized_skin_lesion = sanitize_domain_class(skin_lesion)
    sanitized_risk = sanitize_domain_class(risk)

    if sanitized_skin_lesion in skin_lesion_classes and sanitized_risk in risk_classes:
        return ReportClassificationAnswer(
            valid=True,
            skin_lesion=skin_lesion,
            risk=risk
        )

    return InvalidAnswer(
        valid=False,
        answer=answer
    )


def structure_classification_results(prompt_type: defs.PromptType, results: list[TestResult]) -> list[AnswerResult]:
    '''
    Estrutura as respostas.
    '''

    answer_results = []

    for result in results:
        answer = None

        if prompt_type == defs.PromptType.REPORT:
            answer = structure_report_classification_answer(result.answer)
        else:
            answer = structure_simple_classification_answer(result.answer)

        answer_result = AnswerResult(
            exam_id=result.exam_id,
            image=result.image,
            answer=answer
        )

        answer_results.append(answer_result)

    return answer_results


def associate_simple_results_with_data(dataset: list[SimpleLesionData],
                                       prompt_type: defs.PromptType,
                                       results: list[TestResult]) -> list[SimpleResultPair]:
    '''
    Associa os resultados com os dados.
    '''

    dataset_dict = {lesion_data.exam_id: lesion_data for lesion_data in dataset}
    result_pairs = []

    for result in results:
        if result.exam_id not in dataset_dict:
            continue

        result_pair = SimpleResultPair(
            exam_id=result.exam_id,
            image=result.image,
            answer=result.answer,
            expected=format_answer(prompt_type, dataset_dict[result.exam_id])
        )

        result_pairs.append(result_pair)

    return result_pairs


def associate_classification_results_with_data(dataset: list[SimpleLesionData],
                                               prompt_type: defs.PromptType,
                                               results: list[AnswerResult]) -> list[SimpleClassificationResultPair |
                                                                                    ReportClassificationResultPair]:
    '''
    Associa os resultados com os dados.
    '''

    dataset_dict = {lesion_data.exam_id: lesion_data for lesion_data in dataset}
    result_pairs = []

    if prompt_type == defs.PromptType.REPORT:
        for result in results:
            if result.exam_id not in dataset_dict:
                continue

            report = dataset_dict[result.exam_id].report
            result_answer = None

            if result.answer.valid:
                report_answer: ReportClassificationAnswer = result.answer  # type: ignore

                result_answer = ReportClassification(
                    skin_lesion=report_answer.skin_lesion,
                    risk=report_answer.risk
                )
            else:
                result_answer = ReportClassification(
                    skin_lesion='incerto',
                    risk='incerto'
                )

            expected_answer = ReportClassification(
                skin_lesion=report.skin_lesion,
                risk=report.risk
            )

            result_pair = ReportClassificationResultPair(
                exam_id=result.exam_id,
                image=result.image,
                answer=result_answer,
                expected=expected_answer
            )

            result_pairs.append(result_pair)
    else:
        for result in results:
            if result.exam_id in dataset_dict:
                result_answer = ''

                if result.answer.valid:
                    simple_classification_answer: SimpleClassificationAnswer = result.answer  # type: ignore

                    result_answer = simple_classification_answer.skin_lesion
                else:
                    result_answer = 'incerto'

                expected_answer = dataset_dict[result.exam_id].report.skin_lesion

                result_pair = SimpleClassificationResultPair(
                    exam_id=result.exam_id,
                    image=result.image,
                    answer=result_answer,
                    expected=expected_answer
                )

                result_pairs.append(result_pair)

    return result_pairs


def get_label_pairs(result_pairs: list[SimpleClassificationResultPair | ReportClassificationResultPair],
                    attribute_name: str) -> list[tuple[str, str]]:
    '''
    Obtém os pares de resultados: (Esperado, Previsto).
    '''

    pairs = []

    for result in result_pairs:
        expected = result.expected if isinstance(result.expected, str) else getattr(result.expected, attribute_name)
        answer = result.answer if isinstance(result.answer, str) else getattr(result.answer, attribute_name)

        pairs.append((expected, answer))

    return pairs


def split_pairs(pairs: list[tuple[str, str]],
                classes: dict[str, ClassCount],
                most_frequent_classes_count: int) -> tuple[list[tuple[str, str]],
                                                           list[tuple[str, str]],
                                                           list[str],
                                                           list[str]]:
    '''
    Divide os pares pela frequência das labels
    '''

    class_items = sorted(classes.items(), key=lambda item: item[1].count, reverse=True)
    most_frequent_classes = list(map(lambda x: sanitize_domain_class(x[0]),
                                     class_items[:most_frequent_classes_count]))
    least_frequent_classes = list(map(lambda x: sanitize_domain_class(x[0]),
                                      class_items[most_frequent_classes_count:]))

    most_frequent_label_pairs = []
    least_frequent_label_pairs = []

    for pair in pairs:
        if pair[1] in most_frequent_classes:
            most_frequent_label_pairs.append(pair)
        else:
            least_frequent_label_pairs.append(pair)

    return most_frequent_label_pairs, least_frequent_label_pairs, most_frequent_classes, least_frequent_classes


def calculate_accuracy(pairs: list[tuple[str, str]]) -> float:
    '''
    Calcula a acurácia.
    '''

    y_true = [pair[0] for pair in pairs]
    x_predicted = [pair[1] for pair in pairs]

    return float(accuracy_score(y_true, x_predicted))


def calculate_precision(pairs: list[tuple[str, str]]) -> float:
    '''
    Calcula a precisão.
    '''

    y_true = [pair[0] for pair in pairs]
    x_predicted = [pair[1] for pair in pairs]

    return float(precision_score(y_true, x_predicted, average='weighted', zero_division=0))


def calculate_recall(pairs: list[tuple[str, str]]) -> float:
    '''
    Calcula o recall.
    '''

    y_true = [pair[0] for pair in pairs]
    x_predicted = [pair[1] for pair in pairs]

    return float(recall_score(y_true, x_predicted, average='weighted', zero_division=0))


def calculate_f1(pairs: list[tuple[str, str]]) -> float:
    '''
    Calcula o f1-score.
    '''

    y_true = [pair[0] for pair in pairs]
    x_predicted = [pair[1] for pair in pairs]

    return float(f1_score(y_true, x_predicted, average='weighted', zero_division=0))


def create_confusion_matrix(pairs: list[tuple[str, str]],
                            labels: list[str],
                            title: str,
                            save_path: str,
                            annotate: bool = True,
                            format_: str = '.1%') -> None:
    '''
    Calcula a acurácia e cria uma visualização da matriz de confusão normalizada.
    '''

    y_true = [pair[0] for pair in pairs]
    x_predicted = [pair[1] for pair in pairs]

    accuracy = accuracy_score(y_true, x_predicted)

    cm = confusion_matrix(y_true, x_predicted, labels=labels, normalize='true')

    plt.figure(figsize=(12, 10))
    heat_map = heatmap(cm,
                       annot=annotate,
                       fmt=format_,
                       cmap='Blues',
                       xticklabels=labels,
                       yticklabels=labels,
                       vmin=0,
                       vmax=1)
    color_bar = heat_map.collections[0].colorbar
    color_bar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    color_bar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    plt.title(f'{title}\nAcurácia: {accuracy:.1%}')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_training_loss(event_file_path: str, save_path: str) -> None:
    '''
    Plota a loss de treinamento.
    '''

    events = event_accumulator.EventAccumulator(event_file_path)
    events.Reload()

    train_loss = []
    train_steps = []

    for event in events.Scalars('train/loss'):
        train_steps.append(event.step)
        train_loss.append(event.value)

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label='Loss de treinamento', color='blue')
    plt.title('Loss de treinamento')
    plt.xlabel('Passos')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print('Estatísticas de loss de treinamento:')
    print(pd.Series(train_loss).describe())
