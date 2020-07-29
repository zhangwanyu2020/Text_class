import sys
import random
import math
import numpy as np
import scipy.sparse as sp
import pickle as pkl

word_embedding_dim = 256
word_vectors = {}
doc_name_list = [] # 全部label
doc_train_list = [] # train label
doc_test_list = [] # test label
# baidu_95_bak_label：0	train	高中	生物	分子与细胞	组成细胞的化学元素	组成细胞的化合物
f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/baidu_95_bak_label.txt','r',encoding='utf-8')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split('\t')
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()

doc_content_list = [] # 全部item
f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/baidu_95_bak_item.txt','r',encoding='utf-8')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/train.index','w',encoding='utf-8')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_test_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/test.index','w',encoding='utf-8')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
shuffle_doc_name_list = [] # 包含所有的label，乱序
shuffle_doc_words_list = [] # 包含所有的item，乱序
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/shuffle_label.txt','w',encoding='utf-8')
f.write(shuffle_doc_name_str)
f.close()

f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/shuffle_item.txt','w',encoding='utf-8')
f.write(shuffle_doc_words_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)
word_vectors = np.random.uniform(-0.01,0.01,(vocab_size,word_embedding_dim))
# word_doc_list：{‘科技’:[22，23，29，30000，300000，29999，10000]}
word_doc_list = {}
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] =  [i]
            appeared.add(word)
# word_doc_freq:{科技:7}
word_doc_freq = {}
for word,doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)
# word_id_map:{的:0,我:1,是:2,...}
word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/vocab.txt','w',encoding='utf-8')
f.write(vocab_str)
f.close()

####################################

label_set = set() # 95个
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    for i in range(2,len(temp)):
        label_set.add(temp[i])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/_labels.txt','w',encoding='utf-8')
f.write(label_list_str)
f.close()

######################################

train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size

real_train_doc_names = shuffle_doc_words_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)
f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/real_train.name','w',encoding='utf-8')
f.write(real_train_doc_names_str)
f.close()

######################################

row_x = []
col_x = []
data_x = []

for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embedding_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        id = word_id_map[word] # 找到词在字典对应的id
        word_vector = word_vectors[id]
        doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embedding_dim):
        row_x.append(i)
        col_x.append(j)
        data_x.append(doc_vec[j]/doc_len)

x = sp.csr_matrix((data_x,(row_x,col_x)),shape=(real_train_size,word_embedding_dim))

y = [] # y.shape = [real_train_size,95]
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    one_hot = [0 for l in range(len(label_list))] # label_list 95类
    for label in temp[2:]:
        label_index = label_list.index(label)
        one_hot[label_index] =1
    y.append(one_hot)
y = np.array(y)

test_size = len(test_ids)
row_tx = []
col_tx = []
data_tx = []

for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embedding_dim)])
    doc_words = shuffle_doc_words_list[i+train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        id = word_id_map[word]
        word_vector = word_vectors[id]
        doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embedding_dim):
        row_tx.append(i)
        col_tx.append(j)
        data_tx.append(doc_vec[j]/doc_len)

tx = sp.csr_matrix((data_tx,(row_tx,col_tx)),shape=(test_size,word_embedding_dim))

ty = [] # ty.shape = [test_size,95]
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i+train_size]
    temp = doc_meta.split('\t')
    one_hot = [0 for l in range(len(label_list))]
    for label in temp[2:]:
        label_index = label_list.index(label)
        one_hot[label_index] =1
    ty.append(one_hot)
ty = np.array(ty)

#######################################
# real_train_size 和 test 合并
row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embedding_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        id = word_id_map[word]
        word_vector = word_vectors[id]
        doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embedding_dim):
        row_allx.append(i)
        col_allx.append(j)
        data_allx.append(doc_vec[j]/doc_len)

for i in range(vocab_size):
    for j in range(word_embedding_dim):
        row_allx.append(int(i + train_size))  #这里添加的词是基于train上面再叠加的。
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embedding_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    one_hot = [0 for l in range(len(label_list))]
    for label in temp[2:]:
        label_index = label_list.index(label)
        one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    #如果想该word的label，就在这里该，但是通过实验发现，改动之后不影响实验结果
    ally.append(one_hot)

ally = np.array(ally)
# (18288, 256) (18288, 95) (2257, 256) (2257, 95) (69440, 256) (69440, 95)
print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
#完成了所有feature的提取

#########################################

window_size = 20
windows = [] # windows = [[window_1],[window_2],...[window_n]]

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)


word_window_freq = {} # word_window_freq = {的:20,是:18-->('是'在多少个window中出现)}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {} # word_pair_count = {'0，1':10-->('的'和'是'共现的次数)}  word_pair_count>>word_window_freq
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights
num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = math.log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)  #预留了trainsize的位置呢？？？？？？
    col.append(train_size + j)  #和前面的feature保持一致，先计算文档节点，再计算词和词之间的领结关系
    weight.append(pmi)

# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# doc word frequency
doc_word_freq = {} # doc_word_freq = {'doc:word':文档和词共现的频次} 如果一篇文档中同一个词出现多次，频次也多次

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = math.log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects
dataset = 'baidu_95'
f = open("/Users/zhangwanyu/Desktop/data_project_2/GCN/data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("/Users/zhangwanyu/Desktop/data_project_2/GCN/data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("/Users/zhangwanyu/Desktop/data_project_2/GCN/data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("/Users/zhangwanyu/Desktop/data_project_2/GCN/data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("/Users/zhangwanyu/Desktop/data_project_2/GCN/data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("/Users/zhangwanyu/Desktop/data_project_2/GCN/data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("/Users/zhangwanyu/Desktop/data_project_2/GCN/data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()







