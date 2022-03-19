import os
import datasets

from transformers import AlbertConfig
from transformers import AlbertForMaskedLM
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
from transformers.integrations import TensorBoardCallback
from transformers import pipeline
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

data_path = 'data/trainable_corpus'
output_dir = 'results/v2'
tk_path = 'resources/tokenizer'
dataset_path = 'data/dataset_v1'
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
tokenizer = AutoTokenizer.from_pretrained(tk_path)

data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataset = datasets.load_from_disk(dataset_path)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
tb_writer = SummaryWriter(os.path.join(output_dir, 'logs'))
tb_callback = TensorBoardCallback(tb_writer)

training_args = TrainingArguments(
    output_dir=output_dir,
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
    data_collator=data_collator,
    callbacks=[tb_callback]
)

trainer.train()
trainer.save_model(output_dir)
tb_writer.close()

fill_mask = pipeline(
    "fill-mask",
    model=output_dir,
    tokenizer=tokenizer
)
print(fill_mask("나는 [MASK]를 먹었다."))