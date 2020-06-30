import csv
import os
import pickle
import jieba
import pandas as pd
import fasttext


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


def get_traindata(data_path, train_data_path, train_x_path, test_data_path, stopwords_path):
    """
    data_path: 数据存储路径.csv
    train_data_path: 训练集存储路径
    test_data_path: 测试集存储路径
    stopwords_path: 加载停用词
    """
    data = pd.read_csv(data_path, delimiter=',')
    data.columns = [['Class', 'Question']]
    stopwords = read_file(stopwords_path)
    i, j, k = 0, 0, 0
    for row in data.iterrows():
        line = row[1][0].strip().split(' ')
        if len(line) > 4: k += 1
        Class = line
        # Class = list(set(Class))
        Class = ''.join(Class)
        Question = row[1][1]
        seg_words1 = segment(Question, cut_type='char', pos=False)
        seg_words2 = [word.strip() for word in seg_words1 if word not in stopwords]
        Question = ' '.join(seg_words2)
        line = '__label__' + Class + ' ' + Question
        if i % 10 == 0 and j < 2000:
            with open(test_data_path, 'a', newline='') as f1:
                writer1 = csv.writer(f1)
                writer1.writerow([line])
                i += 1
                j += 1
        else:
            with open(train_data_path, 'a', newline='') as f1:
                writer1 = csv.writer(f1)
                writer1.writerow([line])
            with open(train_x_path, 'a', newline='') as f2:
                writer2 = csv.writer(f2)
                writer2.writerow([Question])
                i += 1
    print('length of train data is {}'.format(i - j))
    print('length of test data is {}'.format(j))
    print('muti_level>4 samples is {}'.format(k))

if __name__ == '__main__':
    get_traindata('/Users/zhangwanyu/Desktop/NLP课程/NLP_week_8/pt20200621/datasets/baidu_95_bak.csv',
                  '/Users/zhangwanyu/Desktop/data_project_2/fasttext_demo/train_data_char.txt',
                  '/Users/zhangwanyu/Desktop/data_project_2/fasttext_demo/train_x_char.txt',
                  '/Users/zhangwanyu/Desktop/data_project_2/fasttext_demo/test_data_char.txt',
                  '/Users/zhangwanyu/stop_words_2.txt'
                  )

    classifier = fasttext.train_supervised("/Users/zhangwanyu/Desktop/data_project_2/fasttext_demo/train_data_char.txt",
                                           label_prefix="__label__")

    classifier.save_model("/Users/zhangwanyu/Desktop/data_project_2/fasttext_demo/fasttext.model2.bin")

    classifier = fasttext.load_model("/Users/zhangwanyu/Desktop/data_project_2/fasttext_demo/fasttext.model2.bin")

    result = classifier.test("/Users/zhangwanyu/Desktop/data_project_2/fasttext_demo/test_data_char.txt")

    print(result)

    classifier.predict('中 央 官 制 三 公 九 卿 制 郡 县 制 重 农 抑 商 政 策')