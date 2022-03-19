import os
import datasets

from transformers import AutoTokenizer


def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

data_path = 'data/trainable_corpus'
dataset_path = 'data/dataset/v1'
tk_path = 'resources/tokenizer'

data_files = [os.path.join(data_path, path) for path in os.listdir(data_path) if not path.startswith('.')]
tokenizer = AutoTokenizer.from_pretrained(tk_path)
dataset = datasets.load_dataset('text', data_files=data_files, split='train')
dataset = dataset.map(encode, batched=True, batch_size=1000, num_proc=8)
dataset = dataset.remove_columns('text')
dataset.save_to_disk(dataset_path)