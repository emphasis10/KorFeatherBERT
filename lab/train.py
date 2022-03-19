import os
import json
import datasets
import argparse

from transformers import AlbertConfig
from transformers import AlbertForMaskedLM
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
from transformers.integrations import TensorBoardCallback
from transformers import pipeline
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)

tk_path = config['path']['tokenizer']
dataset_path = config['path']['data']
result_path = config['path']['result']

model_config = AlbertConfig(
    vocab_size=config['model']['vocab_size'],
    embedding_size=config['model']['embedding_size'],
    hidden_size=config['model']['hidden_size'],
    num_hidden_layers=config['model']['num_hidden_layers'],
    num_hidden_groups=1,
    num_attention_heads=config['model']['num_attention_heads'],
    intermediate_size=config['model']['intermediate_size'],
    inner_group_num=1,
    hidden_act='gelu_new',
    hidden_dropout_prob=0,
    attention_probs_dropout_prob=0,
    max_position_embeddings=config['model']['max_position_embeddings'],
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    classifier_dropout_prob=0.1,
    position_embedding_type='absolute'
)

model = AlbertForMaskedLM(config=model_config)
tokenizer = AutoTokenizer.from_pretrained(tk_path)

data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataset = datasets.load_from_disk(dataset_path)
os.makedirs(result_path, exist_ok=True)
os.makedirs(os.path.join(result_path, 'logs'), exist_ok=True)
tb_writer = SummaryWriter(os.path.join(result_path, 'logs'))
tb_callback = TensorBoardCallback(tb_writer)

training_args = TrainingArguments(
    output_dir=result_path,
    overwrite_output_dir=True,
    num_train_epochs=config['train']['num_train_epochs'],
    per_device_train_batch_size=config['train']['per_device_train_batch_size'],
    save_steps=config['train']['save_steps'],
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=config['train']['fp16'],
    gradient_accumulation_steps=config['train']['gradient_accumulation_steps'],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    callbacks=[tb_callback]
)

trainer.train()
trainer.save_model(result_path)
tb_writer.close()

fill_mask = pipeline(
    "fill-mask",
    model=result_path,
    tokenizer=tokenizer
)
print(fill_mask("나는 [MASK]를 먹었다."))