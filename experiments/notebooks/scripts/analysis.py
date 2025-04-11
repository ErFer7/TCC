'''
Análise de resultados.
'''

from unicodedata import normalize, combining

from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix
from seaborn import heatmap
from pydantic import BaseModel
from tensorboard.backend.event_processing import event_accumulator

import matplotlib.pyplot as plt
import pandas as pd

from scripts.test import TestResult
from scripts.data import SimpleLesionData

import scripts.definitions as defs


class Answer(BaseModel):
    '''
    Resposta.
    '''

    valid: bool


class ReportPrediction(BaseModel):
    '''
    Classificação do laudo. Inclui apenas os campos relevantes para o estudo e de classificação.
    '''

    skin_lesion: str
    risk: str
    skin_lesion_conclusion: str
    conclusion: str


class SimpleClassificationAnswer(Answer):
    '''
    Resposta da classificação simples.
    '''

    skin_lesion: str


class ReportAnswer(Answer, ReportPrediction):
    '''
    Resposta do laudo.
    '''

    pass


class InvalidAnswer(Answer):
    '''
    Resposta inválida.
    '''

    answer: str


class SanitizedResult(BaseModel):
    '''
    Resultado sanitizado.
    '''

    exam_id: int
    image: str
    answer: SimpleClassificationAnswer | ReportAnswer | InvalidAnswer


class SimpleClassificationResultPair(BaseModel):
    '''
    Par de resultados da classificação simples.
    '''

    exam_id: int
    image: str
    answer: str
    expected: str


class ReportResultPair(BaseModel):
    '''
    Par de resultados do laudo.
    '''

    exam_id: int
    image: str
    answer: ReportPrediction
    expected: ReportPrediction


class TestAnalysis(BaseModel):
    '''
    Análise de teste.
    '''

    test_name: str
    valid_results_on_test_data: list[SanitizedResult]
    valid_results_on_training_data: list[SanitizedResult]
    invalid_results_on_test_data: list[SanitizedResult]
    invalid_results_on_training_data: list[SanitizedResult]


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

    answer = answer.strip()
    answer = sanitize_domain_class(answer)

    valid = False

    # TODO: Melhorar a validação
    for class_ in skin_lesion_classes:
        if class_ in answer:
            valid = True
            answer = class_
            break

    return SimpleClassificationAnswer(
        valid=valid,
        skin_lesion=answer
    )


def structure_report_answer(answer: str) -> ReportAnswer | InvalidAnswer:
    '''
    Estrutura a resposta.
    '''

    valid = True
    answer_parts = answer.splitlines()
    contents = []

    for part in answer_parts:
        label_content_pair = part.split(':', 1)

        if len(label_content_pair) >= 2:
            contents.append(label_content_pair[1].strip())

    skin_lesion = contents.pop(0) if len(contents) > 0 else ''
    risk = contents.pop(0) if len(contents) > 0 else ''
    skin_lesion_conclusions = contents.pop(0) if len(contents) > 0 else ''
    conclusion = contents.pop(0) if len(contents) > 0 else ''

    value_domain_pairs = [
        [skin_lesion, skin_lesion_classes],
        [risk, risk_classes]
    ]

    for i, (content_section, domain) in enumerate(value_domain_pairs):
        if content_section != '':
            value_domain_pairs[i][0] = sanitize_domain_class(content_section)

            if value_domain_pairs[i][0] not in domain:
                valid = False
                break
        else:
            valid = False
            break

    skin_lesion = value_domain_pairs[0][0]
    risk = value_domain_pairs[1][0]

    if valid:
        return ReportAnswer(
            valid=valid,
            skin_lesion=skin_lesion,
            risk=risk,
            skin_lesion_conclusion=skin_lesion_conclusions,
            conclusion=conclusion
        )
    return InvalidAnswer(
        valid=valid,
        answer=answer
    )


def structure_answers(prompt_type: defs.PromptType, results: list[TestResult]) -> list[SanitizedResult]:
    '''
    Estrutura as respostas.
    '''

    sanitized_results = []

    for result in results:
        answer = None

        if prompt_type == defs.PromptType.REPORT:
            answer = structure_report_answer(result.answer)
        else:
            answer = structure_simple_classification_answer(result.answer)

        sanitized_result = SanitizedResult(
            exam_id=result.exam_id,
            image=result.image,
            answer=answer
        )

        sanitized_results.append(sanitized_result)

    return sanitized_results


def associate_results_with_data(dataset: list[SimpleLesionData],
                                prompt_type: defs.PromptType,
                                results: list[SanitizedResult]) -> list[SimpleClassificationResultPair |
                                                                        ReportResultPair]:
    '''
    Associa os resultados com os dados.
    '''

    dataset_dict = {lesion_data.exam_id: lesion_data for lesion_data in dataset}
    result_pairs = []

    if prompt_type == defs.PromptType.REPORT:
        for result in results:
            result_answer = None

            if result.answer.valid:
                report_answer: ReportAnswer = result.answer  # type: ignore

                result_answer = ReportPrediction(
                    skin_lesion=report_answer.skin_lesion,
                    risk=report_answer.risk,
                    skin_lesion_conclusion=report_answer.skin_lesion_conclusion,
                    conclusion=report_answer.conclusion
                )
            else:
                result_answer = ReportPrediction(
                    skin_lesion='incerto',
                    risk='incerto',
                    skin_lesion_conclusion='',
                    conclusion=''
                )

            report = dataset_dict[result.exam_id].report

            expected_answer = ReportPrediction(
                skin_lesion=report.skin_lesion,
                risk=report.risk,
                skin_lesion_conclusion=', '.join(report.skin_lesion_conclusion),
                conclusion=report.conclusion
            )

            result_pair = ReportResultPair(
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

                expected_answer = sanitize_domain_class(dataset_dict[result.exam_id].report.skin_lesion)

                result_pair = SimpleClassificationResultPair(
                    exam_id=result.exam_id,
                    image=result.image,
                    answer=result_answer,
                    expected=expected_answer
                )

                result_pairs.append(result_pair)

    return result_pairs


def get_label_pairs(result_pairs: list[SimpleClassificationResultPair | ReportResultPair],
                    attribute_name: str) -> list[tuple[str, str]] | list[tuple[list[str], list[str]]]:
    '''
    Obtém os pares de resultados. Os pares podem ser multilabel ou não.
    '''

    pairs = []

    for result in result_pairs:
        answer = result.answer if isinstance(result.answer, str) else getattr(result.answer, attribute_name)
        expected = result.expected if isinstance(result.expected, str) else getattr(result.expected, attribute_name)

        if isinstance(answer, list):
            answer = sorted(answer)

        if isinstance(expected, list):
            expected = sorted(expected)

        pairs.append((answer, expected))

    return pairs


def create_confusion_matrix(pairs: list[tuple[str, str]], labels: list[str], title: str, save_path: str):
    '''
    Calcula a acurácia e cria uma visualização da matriz de confusão normalizada.
    '''

    y_true = [pair[0] for pair in pairs]
    x_predicted = [pair[1] for pair in pairs]

    accuracy = accuracy_score(y_true, x_predicted)

    cm = confusion_matrix(y_true, x_predicted, labels=labels, normalize='true')

    plt.figure(figsize=(12, 10))
    heat_map = heatmap(cm,
                       annot=True,
                       fmt='.1%',
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

    return accuracy


# TODO: Refatorar
def create_multilabel_confusion_matrix(pairs: list[tuple[tuple[str], tuple[str]]],
                                       labels: list[str],
                                       title: str,
                                       save_path: str):
    '''
    Calcula a acurácia e cria uma visualização da matriz de confusão multilabel normalizada.
    '''

    y_true = [pair[0] for pair in pairs]
    x_predicted = [pair[1] for pair in pairs]

    accuracy = accuracy_score(y_true, x_predicted)

    cm = multilabel_confusion_matrix(y_true, x_predicted, labels=labels)

    # Configure the subplot grid for multiple confusion matrices
    n_labels = len(labels)
    n_cols = min(3, n_labels)
    n_rows = (n_labels + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_labels > 1 else [axes]

    # Create a confusion matrix for each label
    for i, (label, matrix) in enumerate(zip(labels, cm)):
        # Each multilabel confusion matrix has format:
        # [[TN, FP], [FN, TP]]
        # We need to normalize to get percentages
        total = matrix.sum()
        if total > 0:
            matrix_normalized = matrix / total
        else:
            matrix_normalized = matrix

        # Create heatmap for this label
        heat_map = heatmap(matrix_normalized,
                           annot=True,
                           fmt='.1%',
                           cmap='Blues',
                           xticklabels=['Negative', 'Positive'],
                           yticklabels=['Negative', 'Positive'],
                           ax=axes[i],
                           vmin=0,
                           vmax=1)

        axes[i].set_title(f'Classe: {label}')
        axes[i].set_xlabel('Previsto')
        axes[i].set_ylabel('Verdadeiro')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add overall title and adjust layout
    fig.suptitle(f'{title}\nAcurácia: {accuracy:.2%}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    plt.savefig(save_path, dpi=300)
    plt.show()

    return accuracy


def plot_training_loss(event_file_path: str, save_path: str):
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
