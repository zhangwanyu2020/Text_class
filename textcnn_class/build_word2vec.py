from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors

def build_w2v(path_corpus, path_vocab, w2v_model_path='w2v.model', min_count=0,vocab_size=50000, embedding_size=512):
    w2v = Word2Vec(sentences=LineSentence(path_corpus), size=512, window=5, min_count=min_count, iter=5)
    w2v.save(w2v_model_path)

    model = Word2Vec.load(w2v_model_path)
    model = KeyedVectors.load(w2v_model_path)

    words_vectors = {}
    for word in model.wv.vocab:
        words_vectors[word] = model[word]
    #vocab_dict从1开始标记，embedding_matrix需减1
    vocab_dict = open(path_vocab, encoding='utf-8').readlines()
    embedding_matrix = np.zeros((vocab_size, embedding_size))

    for line in vocab_dict[:vocab_size]:
        word_id = line.split()
        word, i = word_id
        embedding_vector = words_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[int(i)-1] = embedding_vector

    return embedding_matrix