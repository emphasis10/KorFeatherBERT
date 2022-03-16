import os
import datasets

from transformers import AlbertConfig
from transformers import AlbertForMaskedLM
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

data_path = './korean/'
data_files = [os.path.join(data_path, path) for path in os.listdir(data_path)]

config = AlbertConfig(
    vocab_size=10000,
    embedding_size=128,
    hidden_size=768,
    num_hidden_layers=3,
    num_hidden_groups=1,
    num_attention_heads=12,
    intermediate_size=768*4,
    inner_group_num=1,
    hidden_act='gelu_new',
    hidden_dropout_prob=0,
    attention_probs_dropout_prob=0,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    classifier_dropout_prob=0.1,
    position_embedding_type='absolute'
)

model = AlbertForMaskedLM(config=config)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file='tk/tokenizer.json',
    model_max_length=512,
    unk_token='[UNK]',
    sep_token='[SEP]',
    pad_token='[PAD]',
    cls_token='[CLS]',
    mask_token='[MASK]'
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataset = datasets.load_from_disk('dataset_v2')
os.makedirs("results", exist_ok=True)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,
    gradient_accumulation_steps=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()
