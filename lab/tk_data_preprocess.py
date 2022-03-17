import re
import os
import mecab

raw_data_dir = 'text'
tk_data_dir = 'preprocessed'
m = mecab.MeCab()
os.makedirs(tk_data_dir, exist_ok=True)

for i in os.listdir(raw_data_dir):
    os.makedirs(os.path.join(tk_data_dir, i), exist_ok=True)
    for j in os.listdir(os.path.join(raw_data_dir, i)):
        with open(os.path.join(raw_data_dir, i, j), 'r') as f:
            fw = open(os.path.join(tk_data_dir, i, j), 'w')
            for line in f.readlines():  
                if not line:
                    continue
                line = line.replace('\n\n', '\n').replace('&lt;', '').replace('&gt;', '')
                line = re.sub('<.*?>', '', line)
                line = re.sub('\(.*?\)', '', line)
                line = ' '.join(m.morphs(line))
                fw.write(line + '\n')

            f.close()
        
os.chdir(tk_data_dir)
for i in os.listdir():
    os.chdir(i)
    os.system('cat wiki_* > merged_' + i)
    os.system('mv merged_'+i+' ../')
    os.chdir('../')
    os.system('rm -rf '+i)