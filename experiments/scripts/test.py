"""
Análise de modelos
"""

from os import makedirs
from os.path import join, exists
from datetime import datetime
from json import load
from argparse import ArgumentParser
from base64 import b64encode
from ollama import Client

HARD_PROMPT = 'Identify if the image is showing a benign or malignant skin cancer. Use strictly and only the ' + \
              'following outputs without any extra text or special characters, even if you don\'t have a clear ' + \
              'answer you must use one of these classifications: Benign, Malignant'

SOFT_PROMPT = 'Identify if the image is showing a benign or malignant skin cancer. Say that it is not clear if ' + \
              'an indentification is absolutely impossible. Use strictly and only the following outputs without ' + \
              'any extra text or special characters: Benign, Malignant, Not clear'

OLLAMA_URL = 'http://localhost:11434'

parser = ArgumentParser(description='Teste de modelos')
client = Client(OLLAMA_URL)

parser.add_argument('model_name', type=str, help='Modelo a ser testado')
parser.add_argument('--samples', type=int, help='Número de amostras a serem testadas', default=None)
parser.add_argument('--hard', action='store_true', help='Força a escolha entre apenas duas opções', default=False)
parser.add_argument('--writerate', type=int, help='Frequência de escrita do output', default=128)

args = parser.parse_args()

with open(join('data', 'basic_test_dataset.json'), 'r', encoding='utf-8') as file:
    data = load(file)[:args.samples]

prompt = HARD_PROMPT if args.hard else SOFT_PROMPT
total = len(data)
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
test_name = f'{args.model_name}_analysis_{current_time}'
test_output = f'# Image id, Expected answer, Actual answer. size={total}, hard={args.hard}, time={current_time}\n'

directory = join('data', 'analysis')

if not exists(directory):
    makedirs(directory)

print(f'Testando {total} amostras com o modelo {args.model_name}')

try:
    for i, value in enumerate(data):
        image_name = value['image']
        expected = value['result']

        with open(join('data', 'images', image_name), 'rb') as file:
            image = b64encode(file.read()).decode('utf-8')

        answer = client.generate(args.model_name, prompt, images=[image])
        striped_answer = answer['response'].strip()

        test_output += str((image_name, expected, striped_answer)) + '\n'

        if (i + 1) % args.writerate == 0:
            with open(join('data', 'analysis', f'{test_name}.txt'), 'a', encoding='utf-8') as file:
                file.write(test_output)

            test_output = ''

        print(f'[{(i + 1) / total * 100.0:.3f}%] Esperado: {expected}, Resposta: {striped_answer}')
except KeyboardInterrupt:
    print('Teste interrompido!')
    test_output += '# Teste interrompido!\n'

test_output += f'# Teste finalizado em {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'

if len(test_output) > 0:
    with open(join('data', 'analysis', f'{test_name}.txt'), 'a', encoding='utf-8') as file:
        file.write(test_output)

print(f'Teste concluído. Resultados salvos em data/analysis/{test_name}.txt')
