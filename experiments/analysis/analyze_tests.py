"""
Análise de modelos
"""

from os import makedirs
from os.path import join, exists
from json import load
from argparse import ArgumentParser
from base64 import b64encode

parser = ArgumentParser(description='Análise de modelos')

parser.add_argument('files', type=str, nargs='+')

args = parser.parse_args()

with open(join('data', 'analysis_dataset.json'), 'r', encoding='utf-8') as file:
    data = load(file)[:args.samples]

tests = {test: [] for test in args.files}

# prompt = HARD_PROMPT if args.hard else SOFT_PROMPT

# # (expected, actual): count
# answers = {('Benign', 'Benign'): 0,
#            ('Benign', 'Malignant'): 0,
#            ('Malignant', 'Benign'): 0,
#            ('Malignant', 'Malignant'): 0}

# if not args.hard:
#     answers[('Benign', 'Not clear')] = 0
#     answers[('Malignant', 'Not clear')] = 0

# out_of_scope_answers = []
# current_total = 0
# total = len(data)

# print(f'Analisando {total} amostras com o modelo {args.model_name}')

# try:
#     for i, value in enumerate(data):
#         image = value['image']
#         expected = value['result']

#         with open(join('data', 'images', image), 'rb') as file:
#             image = b64encode(file.read()).decode('utf-8')

#         answer = client.generate(args.model_name, prompt, images=[image])
#         response = answer['response'].strip()

#         try:
#             answers[(expected, response)] += 1
#         except KeyError:
#             out_of_scope_answers.append((expected, response))

#         current_total += 1

#         print(f'[{(i + 1) / total * 100.0:.3f}%] Esperado: {expected}, Resposta: {response}')
# except KeyboardInterrupt:
#     print('Análise interrompida')

# benign_benign_answers = answers[("Benign", "Benign")]
# benign_malignant_answers = answers[("Benign", "Malignant")]
# malignant_benign_answers = answers[("Malignant", "Benign")]
# malignant_malignant_answers = answers[("Malignant", "Malignant")]

# benign_benign_answers_rate = benign_benign_answers / current_total * 100.0
# benign_malignant_answers_rate = benign_malignant_answers / current_total * 100.0
# malignant_benign_answers_rate = malignant_benign_answers / current_total * 100.0
# malignant_malignant_answers_rate = malignant_malignant_answers / current_total * 100.0

# if not args.hard:
#     benign_not_clear_answers = answers[("Benign", "Not clear")]
#     malignant_not_clear_answers = answers[("Malignant", "Not clear")]
#     benign_not_clear_answers_rate = benign_not_clear_answers / current_total * 100.0
#     malignant_not_clear_answers_rate = malignant_not_clear_answers / current_total * 100.0

# out_of_scope_answers_rate = len(out_of_scope_answers) / current_total * 100.0

# directory = join('data', 'analysis')

# if not exists(directory):
#     makedirs(directory)

# mode = 'w' if args.clear else 'a'

# with open(join('data', 'analysis', f'{args.model_name}_analysis.txt'), mode, encoding='utf-8') as file:
#     file.write(f'Análise do modelo {args.model_name} com {current_total} de {total} amostras analisadas no modo '
#                f'{"HARD" if args.hard else "SOFT"}:\n\nResposta esperada - Resposta recebida\n\n'
#                f'Benign, Benign: {benign_benign_answers}, {benign_benign_answers_rate:.3f}%\n'
#                f'Benign, Malignant: {benign_malignant_answers}, {benign_malignant_answers_rate:.3f}%\n'
#                f'Malignant, Benign: {malignant_benign_answers}, {malignant_benign_answers_rate:.3f}%\n'
#                f'Malignant, Malignant: {malignant_malignant_answers}, {malignant_benign_answers_rate:.3f}%\n')

#     if not args.hard:
#         file.write(f'Benign, Not Clear: {benign_not_clear_answers}, {benign_not_clear_answers_rate:.3f}%\n'
#                    f'Malignant, Not Clear: {malignant_not_clear_answers}, {malignant_not_clear_answers_rate:.3f}%\n')

#     file.write(f'Respostas fora de escopo: {len(out_of_scope_answers)}, {out_of_scope_answers_rate:.3f}%\n'
#                f'Total: {current_total}\n'
#                f'Precisão: {(benign_benign_answers + malignant_malignant_answers) / current_total * 100.0:.3f}%\n'
#                f'Taxa de erro: {(benign_malignant_answers + malignant_benign_answers) / current_total * 100.0:.3f}%\n')

#     file.write('\nRespostas fora de escopo:\n\n')

#     for i, (expected, actual) in enumerate(out_of_scope_answers):
#         file.write(f'{i + 1}. Esperado: {expected}, Resposta: {actual}\n')

#     file.write(f'\n{"-" * 64}\n')

# print(f'Análise concluída. Resultados salvos em data/analysis/{args.model_name}_analysis.txt')
