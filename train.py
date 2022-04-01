import json
import argparse
import os
import datasets

from transformers import AlbertConfig
from transformers import AlbertForMaskedLM
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer, BertTokenizer
from transformers.integrations import TensorBoardCallback
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import RobertaProcessing
from torch.utils.tensorboard import SummaryWriter
from preprocessor import Preprocessor

class KorFeatherBERT:
    def __init__(self, config):
        self.tokenzier = None
        self.config = config

        self.raw_data_path = self.config['path']['raw_data']
        self.tk_data_path = self.config['path']['tk_data']
        self.model_data_path = self.config['path']['model_data']
        self.tk_path = self.config['path']['tokenizer']
        self.result_path = self.config['path']['result']

        self.preprocessor = Preprocessor(self.config)

    def __train_tokenizer(self):
        data_files = []
        for path in os.listdir(self.tk_data_path):
            if not path.startswith('.'):
                data_files.append(os.path.join(self.tk_data_path, path))

        tokenizer = BertWordPieceTokenizer(handle_chinese_chars=False)
        tokenizer.train(files=data_files, vocab_size=10000, min_frequency=2)
        os.makedirs(self.tk_path, exist_ok=True)
        tokenizer.save(os.path.join(self.tk_path, 'tokenizer.json'))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(self.tk_path, 'tokenizer.json'),
            model_max_length=512,
            unk_token='[UNK]',
            sep_token='[SEP]',
            pad_token='[PAD]',
            cls_token='[CLS]',
            mask_token='[MASK]'
        )
        tokenizer.save_pretrained(self.tk_path)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tk_path)

    def __train_model(self):
        model_config = AlbertConfig(
            vocab_size=self.config['model']['vocab_size'],
            embedding_size=self.config['model']['embedding_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_hidden_layers=self.config['model']['num_hidden_layers'],
            num_hidden_groups=1,
            num_attention_heads=self.config['model']['num_attention_heads'],
            intermediate_size=self.config['model']['intermediate_size'],
            inner_group_num=1,
            hidden_act='gelu_new',
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            max_position_embeddings=self.config['model']['max_position_embeddings'],
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            classifier_dropout_prob=0.1,
            position_embedding_type='absolute'
        )

        model = AlbertForMaskedLM(config=model_config)
        tokenizer = AutoTokenizer.from_pretrained(self.tk_path)

        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        dataset = datasets.load_from_disk(self.model_data_path)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, 'logs'), exist_ok=True)
        tb_writer = SummaryWriter(os.path.join(self.result_path, 'logs'))
        tb_callback = TensorBoardCallback(tb_writer)

        training_args = TrainingArguments(
            output_dir=self.result_path,
            overwrite_output_dir=True,
            num_train_epochs=self.config['train']['num_train_epochs'],
            per_device_train_batch_size=self.config['train']['per_device_train_batch_size'],
            save_steps=self.config['train']['save_steps'],
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=self.config['train']['fp16'],
            gradient_accumulation_steps=self.config['train']['gradient_accumulation_steps'],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[tb_callback]
        )

        checkpoint_list = [os.path.join(self.result_path, ck) for ck in os.listdir(self.result_path) if 'checkpoint' in ck]
        checkpoint_list = sorted(checkpoint_list)
        if len(checkpoint_list):
            trainer.train(checkpoint_list[-1])
        else:
            trainer.train()
        trainer.save_model(self.result_path)
        tb_writer.close()

        fill_mask = pipeline(
            "fill-mask",
            model=self.result_path,
            tokenizer=tokenizer
        )
        print(fill_mask("나는 [MASK]를 먹었다."))

    def train_all(self):
        # 1. Check if tokenizer exist
        #   1-1. If not, preprocess data for tokenizer
        #   1-2. Train Tokenizer
        #   1-3. Save Tokenizer
        # 2. Load Tokenizer
        # 3. Check if arrow dataset exist
        #   3-1. If not, preprocess data for model
        #   3-2. Build arrow dataset from text
        # 4. Train model

        if not os.path.isdir(self.tk_path):
            if not self.tk_data_path or not os.path.isdir(self.tk_data_path):
                raise ValueError
            self.preprocessor.preprocess_for_tk()
            self.__train_tokenizer()
        self.tokenzier = self.load_tokenizer()

        if not os.path.isdir(self.model_data_path):
            self.preprocessor.preprocess_for_model(self.tokenzier)
        self.__train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    KorFeatherBERT(config).train_all()