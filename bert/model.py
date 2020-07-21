

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import bert_master.modeling as modeling
from text_loader import TextLoader
import numpy as np


class Project_model():
    def __init__(self,bert_root,data_path,temp_path,model_save_path,batch_size,max_len,lr,keep_prob):
        self.bert_root = bert_root
        self.data_path = data_path
        self.temp_path = temp_path
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.lr = lr
        self.keep_prob = keep_prob
        
        self.bert_config()
        self.get_output()
        self.get_loss(True)
        self.get_accuracy()
        self.get_trainOp()
        self.init_saver()
    
    def bert_config(self):
        bert_config_file = os.path.join(self.bert_root, 'bert_config.json')
        # 获取预训练模型的参数文件
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        self.init_checkpoint = os.path.join(self.bert_root, 'bert_model.ckpt')
        self.bert_vocab_file = os.path.join(self.bert_root, 'vocab.txt')
        #初始化变量
        self.input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
        self.input_y = tf.placeholder(tf.float32, shape=[None, 95, 1], name="input_y")
        self.global_step = tf.Variable(0, trainable=False)
        output_weights = tf.get_variable("output_weights", [768, 95],initializer=tf.contrib.layers.xavier_initializer())
        output_bias = tf.get_variable("output_bias", [95,], initializer=tf.contrib.layers.xavier_initializer())
        self.w_out = output_weights
        self.b_out = output_bias
        ########
        # output_weights2 = tf.get_variable("output_weights2", [256, 95],initializer=tf.contrib.layers.xavier_initializer())
        # output_bias2 = tf.get_variable("output_bias2", [95, ], initializer=tf.contrib.layers.xavier_initializer())
        # self.w_out2 = output_weights2
        # self.b_out2 = output_bias2
        ########
        # 初始化bert model
        model = modeling.BertModel(
                                   config=self.bert_config,
                                   is_training=False,
                                   input_ids=self.input_ids,
                                   input_mask=self.input_mask,
                                   token_type_ids=self.segment_ids,
                                   use_one_hot_embeddings=False)
                                   # 变量赋值
                                   tvars = tf.trainable_variables()
                                   (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
                                   tf.train.init_from_checkpoint(self.init_checkpoint, assignment)
                                   # 这个获取句子的output，shape = 768
                                   output_layer_pooled = model.get_pooled_output()
                                   # 添加dropout层，减轻过拟合
        self.output_layer_pooled = tf.nn.dropout(output_layer_pooled, keep_prob=self.keep_prob)
    # return self.output_layer_pooled
    
    def get_output(self):
        # pred 全连接层 768-->95
        self.pred = tf.add(tf.matmul(self.output_layer_pooled, self.w_out), self.b_out, name="pre1")
        self.probabilities = tf.nn.sigmoid(self.pred, name='probabilities')
        self.pred  = tf.reshape(self.pred , shape=[-1, 95, 1], name='pre')
        return self.pred
    
    def get_loss(self,if_regularization):
        net_loss = tf.square(tf.reshape(self.probabilities, [-1]) - tf.reshape(3*self.input_y, [-1]))
        if if_regularization:
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.w_out)
            tf.add_to_collection(tf.GraphKeys.BIASES, self.b_out)
            regularizer = tf.contrib.layers.l1_regularizer(scale=5.0 / 50000)
            reg_loss = tf.contrib.layers.apply_regularization(regularizer)
            net_loss = net_loss + reg_loss
        self.loss = tf.math.reduce_sum(net_loss)/self.batch_size
        return self.loss
    
    def get_accuracy(self):
        self.predicts = tf.argmax(self.pred, axis=-1)
        self.predicts_bool = tf.math.greater(self.probabilities, tf.constant(0.7))
        self.actuals = tf.argmax(self.input_y, axis=-1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicts, self.actuals), dtype=tf.float32))
    
    def get_trainOp(self):
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        return self.train_op
    
    def evaluate(self,sess,devdata):
        data_loader = TextLoader(devdata, self.batch_size)
        accuracies = []
        predictions = []
        reals =[]
        for i in range(data_loader.num_batches):
            x_train, y_train = data_loader.next_batch(i)
            x_input_ids = x_train[:, 0]
            x_input_mask = x_train[:, 1]
            x_segment_ids = x_train[:, 2]
            feed_dict = {self.input_ids: x_input_ids, self.input_mask: x_input_mask,
                self.segment_ids: x_segment_ids,
                    self.input_y: y_train}
            pre = sess.run(self.predicts_bool, feed_dict=feed_dict)
            predictions.append(pre)
            real = feed_dict[self.input_y]
            real.resize([self.batch_size,95])
            reals.append(real)
        acc = np.mean(accuracies) * 100
        return predictions, reals
    
    
    def run_step(self,sess,x_train,y_train):
        x_input_ids = x_train[:, 0]
        x_input_mask = x_train[:, 1]
        x_segment_ids = x_train[:, 2]
        step, loss_, _ = sess.run([self.global_step, self.loss, self.train_op],
                                  feed_dict={self.input_ids: x_input_ids, self.input_mask: x_input_mask,
                                  self.segment_ids: x_segment_ids,
                                  self.input_y: y_train})
        return step,loss_
    
    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())



