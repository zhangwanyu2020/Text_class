import os
import csv
import jieba
import numpy as np
import pandas as pd


# get classes
def get_class_levels(data_path):
    class_level_1 = os.listdir(data_path)
    class_level_2_union = []
    for path_name in class_level_1:
        class_level_2 = []
        filePath = '{}/{}'.format(data_path, path_name)
        c_1 = []
        c_csv = os.listdir(filePath)
        for i in c_csv:
            c_1.append(i.strip('.csv'))
        class_level_2.extend(c_1)
        class_level_2_union.append(class_level_2)

    return class_level_1, class_level_2_union


# concat(level_1,level_2)
def get_classes(class_level_1, class_level_2):
    class_levels = []
    for i in range(len(class_level_1)):
        level_1 = class_level_1[i]
        for level_2 in class_level_2[i]:
            level = level_1 + '_' + level_2
            class_levels.append(level)
    return class_levels


# get data as txt
def get_data(data_path, save_path):
    dir_level_1 = os.listdir(data_path)

    for dir_1 in dir_level_1:
        c_csv = os.listdir('{}/{}'.format(data_path, dir_1))
        for dir_2 in c_csv:
            filePath = '{}/{}/{}'.format(data_path, dir_1, dir_2)
            class_level = (dir_1 + '_' + dir_2).strip('.csv')
            df = pd.read_csv(filePath, delimiter="\t")
            # 每个文件取10条数据做实验
            df = df[0:40]
            df.columns = ['Question']
            df['Class'] = class_level
            # 第一个文件写入header，其他都不需要
            if not os.path.exists(save_path):
                df.to_csv(save_path, mode='a', index=None)
            else:
                df.to_csv(save_path, mode='a', index=None, header=False)


# read stop words
def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip('\n').split()
            lines.extend(line)
    return lines


# get train_x,train_y,clean data
def clean_data(data_path, stop_words):
    data = pd.read_csv(data_path)
    train_x = []
    train_y = []
    for item in data.iterrows():
        line = item[1]['Question']
        seg_words1 = jieba.lcut(line)
        seg_words2 = [word for word in seg_words1 if word not in stop_words]
        train_x.append(seg_words2)
        train_y.append(item[1]['Class'])
    return train_x, train_y


def get_vocab(train_x):
    vocab_set = set()
    for item in train_x:
        set_item = set(item)
        vocab_set = vocab_set.union(set_item)
    vocab = list(vocab_set)
    return vocab, len(vocab)


def get_onehot(vocab, train_x):
    onehot_matrix = np.zeros((len(train_x), len(vocab)))
    for i in range(len(train_x)):
        for index, word in enumerate(vocab):
            if word in train_x[i]:
                onehot_matrix[i][index] = 1
    return onehot_matrix


def get_class_rates(train_y):
    class_counts = {}
    len_y = len(train_y)
    for i in range(len_y):
        class_level = train_y[i]
        if class_level in class_counts:
            class_counts[class_level] += 1
        else:
            class_counts[class_level] = 1

    return class_counts


# get class rates-of-words
def train(train_x_onehot, train_y, class_counts):
    count_words = {}
    sum_class = {}
    V = 20
    numWords = len(train_x_onehot[0])
    for index in class_counts.keys():
        count_words[index] = np.ones(numWords)
        sum_class[index] = V

    for item in enumerate(zip(train_x_onehot, train_y)):
        count_words[item[1][1]] = count_words[item[1][1]] + item[1][0]
        sum_class[item[1][1]] = sum_class[item[1][1]] + sum(item[1][0])

    class_rates = {}
    for item in enumerate(zip(count_words.items(), sum_class.items())):
        rate = np.log(item[1][0][1] / item[1][1][1])
        class_rates[item[1][0][0]] = rate

    return class_rates


def get_test_onehot(test_path, vocab, stop_words):
    test_x, test_y = clean_data(test_path, stop_words)
    test_x_onehot = get_onehot(vocab, test_x)
    return test_x_onehot, test_y


def classifyBN(test_onehot, class_rates):
    test_onehot = np.array(test_onehot)
    p_of_class = {}
    for level, rates in class_rates.items():
        p = sum(test_onehot * rates) + np.log(1 / 20)
        p_of_class[level] = p

    sorted_p = sorted(p_of_class.items(), key=lambda item: item[1], reverse=True)
    class_max_p = sorted_p[0][0]

    return class_max_p


def test(test_x_onehot, class_rates, test_y):
    correct = 0
    for i in range(len(test_x_onehot)):
        class_max_p = classifyBN(test_x_onehot[i], class_rates)
        y = test_y[i]
        print('预测值:', class_max_p)
        print('实际值:', y)
        if class_max_p == y:
            correct += 1
        else:
            pass
        print('*' * 20)
    print('正确率:', correct / len(test_x_onehot))


def main():
    class_level_1, class_level_2_union = get_class_levels('/Users/zhangwanyu/Desktop/data_project_2/data')
    class_levels = get_classes(class_level_1, class_level_2_union)
    get_data('/Users/zhangwanyu/Desktop/data_project_2/data', '/Users/zhangwanyu/Desktop/data_project_2/datas_40.csv')
    stop_words = read_file('/Users/zhangwanyu/stop_words_2.txt')
    train_x, train_y = clean_data('/Users/zhangwanyu/Desktop/data_project_2/datas_40.csv', stop_words)
    vocab, vocab_size = get_vocab(train_x)
    train_x_onehot = get_onehot(vocab, train_x)
    class_counts = get_class_rates(train_y)
    class_rates = train(train_x_onehot, train_y, class_counts)
    test_x_onehot, test_y = get_test_onehot('/Users/zhangwanyu/Desktop/data_project_2/datas_test.csv', vocab,
                                            stop_words)
    class_max_p = classifyBN(test_x_onehot[0], class_rates)
    test(test_x_onehot, class_rates, test_y)


main()
