import pandas as pd
import jieba
from utils import read_file
df = pd.read_csv('/Users/zhangwanyu/Desktop/data_project_2/GCN/baidu_95_bak.csv',header=None,names=['labels','item'],dtype=str)

df['item'] = df.item.apply(lambda x:list(jieba.cut(x)))
stopwords = read_file('/Users/zhangwanyu/Desktop/data_project_2/GCN/stopwords.txt')
df['item'] = df.item.apply(lambda x:[word for word in x if word not in stopwords])
with open('/Users/zhangwanyu/Desktop/data_project_2/GCN/baidu_95_bak_item.txt','w',encoding='utf-8') as f:
    for line in df.item:
        f.write(' '.join(line)+'\n')

index_split = len(df)*0.9
with open('/Users/zhangwanyu/Desktop/data_project_2/GCN/baidu_95_bak_label.txt','w',encoding='utf-8') as f:
    for index,row in df.iterrows():
        category = 'train' if index <= index_split else 'test'
        label = '\t'.join(row[0].split())
        line = str(index)+'\t'+category+'\t'+label+'\n'
        f.write(line)



