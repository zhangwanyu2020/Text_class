
import jieba
import os
import numpy as np
from jieba import posseg

def get_data(file_x, file_y, origin_data='./textcnn/baidu_95的副本.csv'):
    if not os.path.exists(file_x):
        preprocess(origin_data)

    x = np.load(file_x)
    y = np.load(file_y)

    return x, y


def segment(sentence, cut_type='word', pos=False):
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip('\n').split()
            lines.extend(line)
    return lines
