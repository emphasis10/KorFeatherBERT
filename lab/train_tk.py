import os

from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import RobertaProcessing
from transformers import PreTrainedTokenizerFast


data_path = './data/tk_data'
output_path = './resources/tokenizer'
data_files = [os.path.join(data_path, path) for path in os.listdir(data_path) if not path.startswith('.')]

tokenizer = BertWordPieceTokenizer(handle_chinese_chars=False)
tokenizer.train(files=data_files, vocab_size=10000, min_frequency=2)
os.makedirs(output_path, exist_ok=True)
tokenizer.save(os.path.join(output_path, 'tokenizer.json'))
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(output_path, 'tokenizer.json'),
    model_max_length=512,
    unk_token='[UNK]',
    sep_token='[SEP]',
    pad_token='[PAD]',
    cls_token='[CLS]',
    mask_token='[MASK]'
)
tokenizer.save_pretrained(output_path)