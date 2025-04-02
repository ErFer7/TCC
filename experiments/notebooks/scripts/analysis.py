'''
Análise de resultados.
'''

from unicodedata import normalize, combining

from sklearn.metrics import accuracy_score, confusion_matrix
from seaborn import heatmap

import matplotlib.pyplot as plt

import scripts.definitions as defs


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


def structure_simple_classification_answer(answer: str) -> dict:
    '''
    Estrutura a resposta.
    '''

    answer = answer.strip()
    answer = sanitize_domain_class(answer)

    valid = False

    for class_ in skin_lesion_classes:
        if class_ in answer:
            valid = True
            break

    return {
        'valid': valid,
        'skin_lesion': answer
    }


def structure_report_answer(answer: str) -> dict:
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

    elementary_lesions = contents.pop(0) if len(contents) > 0 else ''
    secondary_lesions = contents.pop(0) if len(contents) > 0 else ''
    coloration = contents.pop(0) if len(contents) > 0 else ''
    morphology = contents.pop(0) if len(contents) > 0 else ''
    size = contents.pop(0) if len(contents) > 0 else ''
    skin_lesion = contents.pop(0) if len(contents) > 0 else ''
    risk = contents.pop(0) if len(contents) > 0 else ''
    conclusion = contents.pop(0) if len(contents) > 0 else ''

    list_domain_pairs = (
        (elementary_lesions, elementary_lesion_classes),
        (secondary_lesions, secondary_lesion_classes),
        (coloration, coloration_classes),
        (morphology, morphology_classes),
    )

    value_domain_pairs = (
        (size, size_classes),
        (skin_lesion, skin_lesion_classes),
        (risk, risk_classes)
    )

    for i, (content_section, domain) in enumerate(list_domain_pairs):
        if content_section == '':
            contents[i] = list(map(sanitize_domain_class, content_section.split(',')))

            for content in contents[i]:
                if content not in domain:
                    valid = False
                    break
        else:
            valid = False
            break

    if valid:
        for i, (content_section, domain) in enumerate(value_domain_pairs):
            if content_section == '':
                contents[i] = sanitize_domain_class(content_section)

                if contents[i] not in domain:
                    valid = False
                    break
            else:
                valid = False
                break

    if valid:
        return {
            'valid': True,
            'elementary_lesions': contents[0],
            'secondary_lesions': contents[1],
            'coloration': contents[2],
            'morphology': contents[3],
            'size': contents[4],
            'skin_lesion': contents[5],
            'risk': contents[6],
            'conclusion': conclusion
        }
    else:
        return {
            'valid': False,
            'content': answer
        }


def structure_answers(test_type: str, results: list) -> list:
    '''
    Estrutura as respostas.
    '''

    sanitized_results = []

    for result in results:
        sanitized_result = {
            'exam_id': result['exam_id'],
            'answer': {},
        }

        if test_type == 'report':
            sanitized_result['answer'] = structure_report_answer(result['answer'])
        else:
            sanitized_result['answer'] = structure_simple_classification_answer(result['answer'])

        sanitized_results.append(sanitized_result)

    return sanitized_results


def create_confusion_matrix(pairs: list[tuple[str, str]], labels: tuple[str], title: str, save_path: str):
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
    plt.title(f'{title}\nAcurácia: {accuracy:.2%}')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    return accuracy
