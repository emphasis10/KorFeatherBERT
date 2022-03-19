import re
import os

from transformers import AutoTokenizer


def preprocess(line):
    line = line.strip()
    line = line.replace('\n\n', '\n').replace('&lt;', '').replace('&gt;', '')
    line = re.sub('<.*?>', '', line)
    line = re.sub('\(.*?\)', '', line)
    hangul = re.compile('[^ ,.!?0-9a-zA-Zㄱ-ㅎ가-힣]')
    line = hangul.sub('', line)
    line = line.strip()
    return line


data_path = './data/text'  # Original data path from wikiextractor
corpus_path = './data/trainable_corpus'
tk_path = './resources/tokenizer'

os.makedirs(corpus_path, exist_ok=True)
input_files = [path for path in os.listdir(data_path)]

tokenizer = AutoTokenizer.from_pretrained(tk_path)

for i in input_files:
    path = os.path.join(data_path, i)
    files = os.listdir(path)
    os.makedirs(os.path.join(corpus_path, i), exist_ok=True)
    for j in files:
        with open(os.path.join(data_path, i, j)) as fr:
            with open(os.path.join(corpus_path, i, j), 'w') as fw:
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

                    if tmp:
                        new_texts.append(tmp.rstrip())

                    fw.write('\n'.join(new_texts) + '\n')

os.chdir(corpus_path)
for i in os.listdir():
    if len(i) > 2:
        continue
    os.chdir(i)
    os.system('cat wiki_* > merged_' + i)
    os.system('mv merged_' + i + ' ../')
    os.chdir('../')
    os.system('rm -rf ' + i)
