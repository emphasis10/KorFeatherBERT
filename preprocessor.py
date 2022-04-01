import os
import re
import multiprocessing
import datasets
# import mecab

from unicodedata import normalize
from transformers import BertTokenizer

class Preprocessor:
    def __init__(self, config):
        self.raw_data_path = config['path']['raw_data']
        self.tk_data_path = config['path']['tk_data']
        self.model_data_path = config['path']['model_data']
        self.__corpus_path = os.path.join(self.model_data_path, 'corpus')
        self.tk_path = config['path']['tokenizer']
        self.result_path = config['path']['result']

        self.max_len = config['model']['max_position_embeddings']

        self.use_morpheme = config['train']['morpheme']
        self.use_subchar = config['train']['subchar']

        # self.m = mecab.MeCab()

    def __to_subchar(self, string):
        return normalize('NFKD', string)

    def __refine(self, line):
        line = line.strip()
        line = line.replace('\n\n', '\n').replace('&lt;', '').replace('&gt;', '').replace('  ', ' ')
        line = re.sub('<.*?>', '', line)
        line = re.sub('\(.*?\)', '', line)
        hangul = re.compile('[^ ,.!?0-9a-zA-Zㄱ-ㅎ가-힣]')
        line = hangul.sub('', line)
        line = line.strip()
        if self.use_morpheme:
            # line = ' '.join(self.m.morphs(line))
            pass

        if self.use_subchar:
            line = self.__to_subchar(line)
        return line

    def __merge_file_tree(self, target):
        cwd = os.getcwd()
        os.chdir(target)
        for i in os.listdir():
            if len(i) > 2:
                continue
            os.chdir(i)
            os.system('cat wiki_* > merged_' + i)
            os.system('mv merged_' + i + ' ../')
            os.chdir('../')
            os.system('rm -rf ' + i)

        os.chdir(cwd)

    def preprocess_for_tk(self):
        os.makedirs(self.tk_data_path, exist_ok=True)

        for i in os.listdir(self.raw_data_path):
            os.makedirs(os.path.join(self.tk_data_path, i), exist_ok=True)
            for j in os.listdir(os.path.join(self.raw_data_path, i)):
                with open(os.path.join(self.raw_data_path, i, j), 'r') as f:
                    fw = open(os.path.join(self.tk_data_path, i, j), 'w')
                    for line in f.readlines():
                        if not line:
                            continue
                        line = self.__refine(line)
                        fw.write(line + '\n')

                    f.close()

        self.__merge_file_tree(self.tk_data_path)

    def process_file(self, tag_name):
        path = os.path.join(self.raw_data_path, tag_name)
        files = os.listdir(path)
        os.makedirs(os.path.join(self.__corpus_path, tag_name), exist_ok=True)
        for j in files:
            with open(os.path.join(self.raw_data_path, tag_name, j)) as fr:
                with open(os.path.join(self.__corpus_path, tag_name, j), 'w') as fw:
                    paragraphs = []
                    paragraph = ""
                    for line in fr.readlines():
                        if '<doc id' in line:
                            continue

                        if '</doc>' in line:
                            paragraphs.append(paragraph)
                            paragraph = ""
                            continue

                        line = self.__refine(line)
                        if not line:
                            continue

                        if line[-1] != '.':
                            line += '.'
                        paragraph += line

                    for paragraph in paragraphs:
                        texts = paragraph.split('.')
                        new_texts = []
                        tmp = ""
                        for text in texts:
                            text = text.strip()
                            if not text:
                                continue
                            tokenized_text = self.tokenizer.tokenize(text)
                            if len(tmp) + len(tokenized_text) > self.max_len - 2:
                                new_texts.append(tmp.rstrip())
                                tmp = ""
                            tmp += text + '. '

                        if tmp and len(tmp) > 100:
                            new_texts.append(tmp.rstrip())

                        for line in new_texts:
                            line = line.replace('\n', '')
                            line += '\n'
                            fw.write(line)

    def preprocess_for_model(self, tokenizer):
        self.tokenizer = tokenizer
        os.makedirs(self.__corpus_path, exist_ok=True)
        input_files = [path for path in os.listdir(self.raw_data_path)]

        pool = multiprocessing.Pool(processes=8)
        pool.map(self.process_file, input_files)
        pool.close()
        pool.join()

        # for file in input_files:
        #     self.process_file(file, tokenizer)

        self.__merge_file_tree(self.__corpus_path)

        def encode(examples):
            return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

        data_files = []
        for path in os.listdir(self.__corpus_path):
            if not path.startswith('.'):
                data_files.append(os.path.join(self.__corpus_path, path))

        dataset = datasets.load_dataset('text', data_files=data_files, split='train')
        dataset = dataset.map(encode, batched=True, batch_size=1000, num_proc=8)
        dataset = dataset.remove_columns('text')
        dataset.save_to_disk(self.model_data_path)
