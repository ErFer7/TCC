"""
Análise de modelos
"""

import matplotlib.pyplot as plt

from os import makedirs
from ast import literal_eval
from os.path import join, exists
from argparse import ArgumentParser

parser = ArgumentParser(description='Análise de modelos')

parser.add_argument('output_file', type=str, help='Arquivo de saída')
parser.add_argument('files', type=str, nargs=2)
parser.add_argument('--clear', action='store_true', help='Limpa o arquivo de análise', default=False)

args = parser.parse_args()

first_test = []
second_test = []

for i, test in enumerate(args.files):
    with open(join(test), 'r', encoding='utf-8') as file:
        lines = file.readlines()

    test_data = list(map(literal_eval, filter(lambda line: not line.startswith('#'), lines)))

    if i == 0:
        first_test = test_data
    else:
        second_test = test_data

print('Calulando exatidão')

# (expected, actual): count
answers = {('Benign', 'Benign'): 0,
           ('Benign', 'Malignant'): 0,
           ('Malignant', 'Benign'): 0,
           ('Malignant', 'Malignant'): 0,
           ('Benign', 'Other'): 0,
           ('Malignant', 'Other'): 0}

for _, expected, actual in first_test + second_test:
    try:
        answers[(expected, actual)] += 1
    except KeyError:
        answers[(expected, 'Other')] += 1

print('Calculando precisão')

differences = 0

if len(first_test) == len(second_test):
    for result_a, result_b in zip(first_test, second_test):
        if result_a[2] != result_b[2]:
            differences += 1
else:
    print('Os testes não possuem o mesmo tamanho!')

total = sum(answers.values())

benign_benign_answers = answers[("Benign", "Benign")]
benign_malignant_answers = answers[("Benign", "Malignant")]
malignant_benign_answers = answers[("Malignant", "Benign")]
malignant_malignant_answers = answers[("Malignant", "Malignant")]
benign_other_answers = answers[("Benign", "Other")]
malignant_other_answers = answers[("Malignant", "Other")]

benign_benign_answers_rate = benign_benign_answers / total * 100.0
benign_malignant_answers_rate = benign_malignant_answers / total * 100.0
malignant_benign_answers_rate = malignant_benign_answers / total * 100.0
malignant_malignant_answers_rate = malignant_malignant_answers / total * 100.0
benign_other_answers_rate = benign_other_answers / total * 100.0
malignant_other_answers_rate = malignant_other_answers / total * 100.0

directory = 'analysis'

if not exists(directory):
    makedirs(directory)

labels = ['Benign, Benign', 'Benign, Malignant', 'Malignant, Benign',
          'Malignant, Malignant', 'Benign, Other', 'Malignant, Other']
sizes = [benign_benign_answers, benign_malignant_answers, malignant_benign_answers,
         malignant_malignant_answers, benign_other_answers, malignant_other_answers]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']

total = sum(sizes)
percentages = [size / total * 100 for size in sizes]

fig, ax = plt.subplots(figsize=(4, 8))
bottom = 0

for i, label in enumerate(labels):
    ax.bar('Distribuição detalhada', percentages[i], bottom=bottom, color=colors[i], label=label)
    bottom += percentages[i]

ax.set_xlabel('Categorias')
ax.set_ylabel('%')
ax.set_title('Distribuição detalhada')
ax.legend()
plt.tight_layout()
plt.savefig(join(directory, f'{args.output_file}_stacked_bar_chart.png'))
plt.close()

summary_labels = ['Respostas corretas', 'Falsos positivos', 'Falsos negativos', 'Respostas incertas']
summary_sizes = [benign_benign_answers + malignant_malignant_answers, benign_malignant_answers,
                 malignant_benign_answers, benign_other_answers + malignant_other_answers]
summary_colors = ['#36AE7C', '#F9D923', '#EB5353', '#c2c2f0']

summary_total = sum(summary_sizes)
summary_percentages = [size / summary_total * 100 for size in summary_sizes]

fig, ax = plt.subplots(figsize=(4, 8))
bottom = 0

for i, summary_label in enumerate(summary_labels):
    ax.bar(
        'Distribuição de Respostas', summary_percentages[i],
        bottom=bottom, color=summary_colors[i],
        label=summary_label)
    bottom += summary_percentages[i]

ax.set_xlabel('Categorias')
ax.set_ylabel('%')
ax.set_title('Distribuição')
ax.legend()
plt.tight_layout()
plt.savefig(join(directory, f'{args.output_file}_summary_stacked_bar_chart.png'))
plt.close()

mode = 'w' if args.clear or not exists(join(directory, f'{args.output_file}.txt')) else 'a'

with open(join(directory, f'{args.output_file}.txt'), mode, encoding='utf-8') as file:
    file.write(f'Análise com {total} amostras.\n\nResposta esperada - Resposta recebida:\n\n'
               f'Benign, Benign: {benign_benign_answers}, {benign_benign_answers_rate:.3f}%\n'
               f'Benign, Malignant: {benign_malignant_answers}, {benign_malignant_answers_rate:.3f}%\n'
               f'Malignant, Benign: {malignant_benign_answers}, {malignant_benign_answers_rate:.3f}%\n'
               f'Malignant, Malignant: {malignant_malignant_answers}, {malignant_benign_answers_rate:.3f}%\n'
               f'Benign, Other: {benign_other_answers}, {benign_other_answers_rate:.3f}%\n'
               f'Malignant, Other: {malignant_other_answers}, {malignant_other_answers_rate:.3f}%\n')

    file.write(f'Total: {total}\n'
               f'Taxa de acertos: {(benign_benign_answers + malignant_malignant_answers) / total * 100.0:.3f}%\n'
               f'Taxa de erro: {(benign_malignant_answers + malignant_benign_answers) / total * 100.0:.3f}%\n'
               f'Taxa de reincidência: {100.0 - differences / total * 100.0:.3f}%\n\n')

    file.write(f'{"-" * 64}\n')


print(f'Análise concluída. Resultados salvos em {directory}/{args.output_file}.txt')
