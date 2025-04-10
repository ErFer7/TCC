'''
Treinamento em múltiplas GPUs
'''

from os.path import join
from json import load, dump
from datetime import timedelta

from unsloth import FastVisionModel
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from tqdm.notebook import tqdm

import torch

from scripts.authentication import authenticate_huggingface
from scripts.data import LesionData, DatasetAnalysis
from scripts.messages import create_training_message
from scripts.training import Training

import scripts.definitions as defs


authenticate_huggingface()

VERSION = '0.2'

training_hyperparameters = Training(
    base_model_name=defs.BASE_MODEL_NAME,
    trained_model_name=defs.MODEL_NAME,
    quantization=True,
    prompt_type=defs.PromptType.REPORT,
    version=VERSION,
    size=11,
    peft_hyperparameters={
        'finetune_vision_layers': True,
        'finetune_language_layers': True,
        'finetune_attention_modules': True,
        'finetune_mlp_modules': True,
        'r': 64,
        'lora_alpha': 64,
        'lora_dropout': 0.05,
        'bias': 'none',
        'random_state': 3407,
        'use_rslora': True,
        'loftq_config': None
    },
    sft_hyperparameters={
        'seed': defs.STATIC_RANDOM_STATE,
        'gradient_accumulation_steps': 4,
        'per_device_train_batch_size': 1,
        'eval_strategy': 'steps',
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'lr_scheduler_type': 'constant',
        'bf16': is_bf16_supported(),
        'fp16': not is_bf16_supported(),
        'remove_unused_columns': False,
        'optim': 'paged_adamw_32bit',
        'report_to': 'tensorboard',
        'logging_steps': 0.1,
        'output_dir': 'outputs',
        'run_name': f'training_{VERSION}',
        'dataset_text_field': '',
        'dataset_kwargs': {'skip_prepare_dataset': True},
        'dataset_num_proc': 4,
        'max_seq_length': 2048
    },
    used_memory=0.0,
    training_time=0
)

with open(join(defs.TRAINING_PATH, f'hyperparameters_{training_hyperparameters.version}.json'), 'w', encoding='utf-8') as file:
    dump(training_hyperparameters.model_dump(), file, indent=4, ensure_ascii=False)

with open(join(defs.DATA_PATH, 'stt_data', 'training_dataset.json'), 'r', encoding='utf-8') as file:
    training_dataset = [LesionData(**data) for data in load(file)]

with open(join(defs.DATA_PATH, 'training_dataset_analysis.json'), 'r', encoding='utf-8') as file:
    training_dataset_analysis = DatasetAnalysis(**load(file))

training_messages = []
validation_messages = []

for lesion_data in tqdm(training_dataset, desc='Criando mensagens de treinamento: '):
    training_messages.append(create_training_message(training_hyperparameters.prompt_type,
                                                     lesion_data,
                                                     training_dataset_analysis))

model, tokenizer = FastVisionModel.from_pretrained(
    training_hyperparameters.base_model_name,
    load_in_4bit=training_hyperparameters.quantization,
    use_gradient_checkpointing='unsloth'
)

model = FastVisionModel.get_peft_model(
    model,
    **training_hyperparameters.peft_hyperparameters
)

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=training_messages,
    args=SFTConfig(**training_hyperparameters.sft_hyperparameters),
)

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f'Tempo de treinamento: {timedelta(seconds=trainer_stats.metrics["train_runtime"])}')
print(f'Memória máxima reservada: {used_memory} GB')

training_hyperparameters.used_memory = used_memory
training_hyperparameters.training_time = trainer_stats.metrics['train_runtime']

with open(join(defs.TRAINING_PATH, f'hyperparameters_{training_hyperparameters.version}.json'), 'w', encoding='utf-8') as file:
    dump(training_hyperparameters.model_dump(), file, indent=4, ensure_ascii=False)

trained_model_name = f'{training_hyperparameters.trained_model_name}-{training_hyperparameters.version}-{training_hyperparameters.size}B'

if training_hyperparameters.quantization:
    trained_model_name += '-4bit'

if training_hyperparameters.prompt_type == defs.PromptType.SIMPLE_CLASSIFICATION:
    trained_model_name += '-SC'

save_path = join(defs.RESULTS_PATH, 'adapter_weights', trained_model_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

with open(join(defs.TRAINING_PATH, 'models.json'), 'r', encoding='utf-8') as file:
    models = {name: defs.Model(**data) for name, data in load(file)}

new_model = defs.Model(
    local=True,
    quantized=training_hyperparameters.quantization,
    prompt_type=training_hyperparameters.prompt_type,
    version=training_hyperparameters.version,
    size=training_hyperparameters.size
)

models[trained_model_name] = new_model

for name, trained_model in models:
    models[name] = trained_model.model_dump()

with open(join(defs.TRAINING_PATH, 'models.json'), 'w', encoding='utf-8') as file:
    dump(models, file, indent=4, ensure_ascii=False)
