import re
import os
import mecab


m = mecab.MeCab()
os.makedirs('preprocessed', exist_ok=True)

for i in os.listdir('text'):
    os.makedirs('preprocessed/'+i, exist_ok=True)
    for j in os.listdir('text/'+i):
        with open('text/'+i+'/'+j, 'r') as f:
            fw = open('preprocessed/'+i+'/'+j, 'w')
            for line in f.readlines():  
                if not line:
                    continue
                line = line.replace('\n\n', '\n').replace('&lt;', '').replace('&gt;', '')
                line = re.sub('<.*?>', '', line)
                line = re.sub('\(.*?\)', '', line)
                line = ' '.join(m.morphs(line))
                fw.write(line + '\n')

            f.close()
        
os.chdir('preprocessed')
for i in os.listdir():
    os.chdir(i)
    os.system('cat wiki_* > merged_' + i)
    os.system('mv merged_'+i+' ../')
    os.chdir('../')
    os.system('rm -rf '+i)    