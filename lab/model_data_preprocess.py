import re
import os
import multiprocessing

from transformers import AutoTokenizer


def preprocess(line):
    line = line.strip()
    line = line.replace('\n\n', '\n').replace('&lt;', '').replace('&gt;', '').replace('  ', ' ')
    line = re.sub('<.*?>', '', line)
    line = re.sub('\(.*?\)', '', line)
    hangul = re.compile('[^ ,.!?0-9a-zA-Zㄱ-ㅎ가-힣]')
    line = hangul.sub('', line)
    line = line.strip()
    return line

def process_file(tag_name):
    path = os.path.join(data_path, tag_name)
    files = os.listdir(path)
    os.makedirs(os.path.join(corpus_path, tag_name), exist_ok=True)
    for j in files:
        with open(os.path.join(data_path, tag_name, j)) as fr:
            with open(os.path.join(corpus_path, tag_name, j), 'w') as fw:
                paragraphs = []
                paragraph = ""
                for line in fr.readlines():
                    if '<doc id' in line:
                        continue

                    if '</doc>' in line:
                        paragraphs.append(paragraph)
                        paragraph = ""
                        continue

                    line = preprocess(line)
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
                        tokenized_text = tokenizer.tokenize(text)
                        if len(tmp) + len(tokenized_text) > 510:
                            new_texts.append(tmp.rstrip())
                            tmp = ""
                        tmp += text + '. '

                    if tmp and len(tmp) > 100:
                        new_texts.append(tmp.rstrip())

                    for line in new_texts:
                        line = line.replace('\n', '')
                        line += '\n'
                        fw.write(line)

data_path = './data/text'  # Original data path from wikiextractor
corpus_path = './data/trainable_corpus'
tk_path = './resources/tokenizer'

os.makedirs(corpus_path, exist_ok=True)
input_files = [path for path in os.listdir(data_path)]

tokenizer = AutoTokenizer.from_pretrained(tk_path)

pool = multiprocessing.Pool(processes=8)
pool.map(process_file, input_files)
pool.close()
pool.join()

os.chdir(corpus_path)
for i in os.listdir():
    if len(i) > 2:
        continue
    os.chdir(i)
    os.system('cat wiki_* > merged_' + i)
    os.system('mv merged_' + i + ' ../')
    os.chdir('../')
    os.system('rm -rf ' + i)
