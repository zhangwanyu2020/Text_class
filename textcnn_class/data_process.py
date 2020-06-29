import pandas as pd
from tensorflow.keras import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

def preprocess(data_path, stopwords, save_dir='./textcnn/', vocab_size=50000, padding_size=256):
               padding_size=256):
    data = pd.read_csv(data_path, header=None, names=['label', 'item'], dtype=str)
    # 将语料的每一行分词并去除停用词
    data['item'] = data['item'].apply(lambda x: [word for word in segment(x) if word not in stopwords])
    # 将每一行语料合成一个列表
    corpus = data['item'].tolist()

    # 写出语料到txt,用于词向量训练
    with open(save_dir + 'corpus.txt', 'w', encoding='utf-8') as f:
        for line in corpus:
            line = ' '.join(line)
            f.write('{}\n'.format(line))

    # 生成Tokenizer对象
    text_preprocesser = preprocessing.text.Tokenizer(num_words=vocab_size)
    # 调用Tokenizer.fit_on_texts的方法
    text_preprocesser.fit_on_texts(corpus)
    # 将文本转化成词id,不含排序在vocab_size之后的词，所以len(x[i])<=len(corpus[i])
    x = text_preprocesser.texts_to_sequences(corpus)

    # word_dict包含所有词汇，Word：id
    word_dict = text_preprocesser.word_index
    # 写出word_dict到txt
    with open(save_dir + 'vocab.txt', 'w', encoding='utf-8') as f:
        for word, index in word_dict.items():
            f.write('{0}\t{1}\n'.format(word, index))

    # 补全&截断
    x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size, padding='post', truncating='post')
    # label按空格分开&去重&转列表
    y = data['label'].apply(lambda x: set(x.split())).tolist()
    # 生成MultiLabelBinarizer对象
    mlb = MultiLabelBinarizer()
    # 调用MultiLabelBinarizer.fit_transform方法
    y = mlb.fit_transform(y)

    # 写出label到txt
    with open(save_dir + 'labels_.txt', 'w', encoding='utf-8') as f:
        for label in mlb.classes_:
            f.write('{}\n'.format(label))

    # 分割训练集&测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

    np.save(f'{save_dir}train_x.npy', train_x)
    np.save(f'{save_dir}test_x.npy', test_x)
    np.save(f'{save_dir}train_y.npy', train_y)
    np.save(f'{save_dir}test_y.npy', test_y)
