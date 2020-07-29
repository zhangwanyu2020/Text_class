from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sklearn import metrics
from utils import *
from model import GCN, MLP
import random
import os
import sys

dataset = 'baidu_95'

# Set random seed
seed = 1 # random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)
# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 90, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 756, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('model_save_path', '/Users/zhangwanyu/Desktop/data_project_2/GCN/model/', 'Model save path')
flags.DEFINE_integer('save_checkpoint_steps', 1000, 'How many steps to save the model')
flags.DEFINE_string('mode', 'test', 'train or test')
# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(FLAGS.dataset)
#adj是一个矩阵， features是X
print('features.shape',features.shape)
print('y_train.shape',y_train.shape)
print('y_val.shape',y_val.shape)
print('y_test.shape',y_test.shape)
features = sp.identity(features.shape[0])  # featureless
# Some preprocessing 实际是做归一化
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
# print(features[2][1])
model = model_func(placeholders, input_dim=features[2][1], multi_label=True, logging=True)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
if FLAGS.mode == 'train':
    saver = tf.train.Saver()
    for epoch in range(FLAGS.epochs):
        t = time.time()
        feed_dict = construct_feed_dict(
            features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([model.opt_op, model.loss, model.accuracy,model.layers[0].embedding], feed_dict=feed_dict)
        saver.save(sess, save_path=FLAGS.model_save_path + 'model.ckpt', global_step=FLAGS.save_checkpoint_steps)
        cost, acc, pred, labels, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        print("Epoch:",'%04d' % (epoch + 1),
            "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]),
            "val_loss=", "{:.5f}".format(cost),
            "val_acc=", "{:.5f}".format(acc),
            "time=", "{:.5f}".format(time.time() - t)
            )

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

print("Optimization Finished!")

if FLAGS.mode == 'test':
    saver = tf.train.Saver()
    model_path = FLAGS.model_save_path + 'model.ckpt-1000'
    saver.restore(sess, model_path)
    test_cost, test_acc, pred, labels, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print('pred.shape: ',pred.shape)
    print("Test set results:",
        "cost=","{:.5f}".format(test_cost),
        "accuracy=","{:.5f}".format(test_acc),
        "time=", "{:.5f}".format(test_duration)
        )

    test_pred = []
    test_labels = []
    print(len(test_mask))
    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(pred[i])
            test_labels.append(labels[i])
    print('pred',test_pred[0])
    print('real',test_labels[0])
    print("Test Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_labels, test_pred, digits=4))
    print("Macro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
    print("Micro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

############################################


# # doc and word embeddings
# print('embeddings:')
# word_embeddings = outs[3][train_size: adj.shape[0] - test_size]
# train_doc_embeddings = outs[3][:train_size]  # include val docs
# test_doc_embeddings = outs[3][adj.shape[0] - test_size:]
#
# print(len(word_embeddings), len(train_doc_embeddings),
#       len(test_doc_embeddings))
# print(word_embeddings)
#
# f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/_labels.txt', 'r')
# words = f.readlines()
# f.close()
#
# vocab_size = len(words)
# word_vectors = []
# for i in range(vocab_size):
#     word = words[i].strip()
#     word_vector = word_embeddings[i]
#     word_vector_str = ' '.join([str(x) for x in word_vector])
#     word_vectors.append(word + ' ' + word_vector_str)
#
# word_embeddings_str = '\n'.join(word_vectors)
# f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/_word_vectors.txt', 'w')
# f.write(word_embeddings_str)
# f.close()
#
# doc_vectors = []
# doc_id = 0
# for i in range(train_size):
#     doc_vector = train_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1
#
# for i in range(test_size):
#     doc_vector = test_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1
#
# doc_embeddings_str = '\n'.join(doc_vectors)
# f = open('/Users/zhangwanyu/Desktop/data_project_2/GCN/_doc_vectors.txt', 'w')
# f.write(doc_embeddings_str)
# f.close()