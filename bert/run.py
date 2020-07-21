
import os
import tensorflow as tf
from model import Project_model
from processor import process_function
from text_loader import TextLoader
import numpy as np


#超参数
epochs = 1
batch_size = 16
max_len = 128
lr = 5e-6  # 学习率
keep_prob = 0.8
bert_root = '/Users/zhangwanyu/Desktop/data_project_2/bert/bert_model_chinese'
bert_vocab_file = os.path.join(bert_root, 'vocab.txt')
model_save_path = '/Users/zhangwanyu/Desktop/data_project_2/bert/model/'
save_checkpoint_steps = 2
#获取数据
data_path = '/Users/zhangwanyu/Desktop/data_project_2/bert/data'
# './temp' 没有用上
train_input,eval_input,predict_input =process_function(data_path,bert_vocab_file,True,True,True,
                                               './temp',max_len,batch_size)
def train():
    model = Project_model(bert_root,data_path,'./temp',model_save_path,batch_size,max_len,lr,keep_prob)
    with tf.Session() as sess:
        # with tf.device('/gpu:0'):
        writer = tf.summary.FileWriter('./tf_log/', sess.graph)
        # saver = tf.train.Saver()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        data_loader = TextLoader(train_input,batch_size)
        step = 0
        for i in range(epochs):
            data_loader.shuff()
            for j in range(data_loader.num_batches):
                x_train,y_train = data_loader.next_batch(j)
                # print(y_train.shape)
                # print(y_train)
                _, loss_= model.run_step(sess,x_train,y_train)
                step += 1
                saver = tf.train.Saver(tf.global_variables())
                saver.save(sess, save_path=model_save_path)
                print('current step is : {}'.format(step))
                print('the epoch number is : %d the index of batch is :%d, the loss value is :%f'%(i, j, loss_))



if __name__ == '__main__':
    train()


