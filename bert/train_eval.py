
import os
import pandas as pd
import tensorflow as tf
from model import Project_model
from processor import process_function
from text_loader import TextLoader
from evaluate import bool_to_value,id_to_labeltext,f1_np
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
save_checkpoint_steps = 100
data_path = '/Users/zhangwanyu/Desktop/data_project_2/bert/data_small'
mode = 'train'

# './temp' 没有用上
train_input,eval_input,predict_input =process_function(data_path,bert_vocab_file,True,True,False,
                                                       './temp',max_len,batch_size)
model = Project_model(bert_root,data_path,'./temp',model_save_path,batch_size,max_len,lr,keep_prob)


if mode == 'train':
    
    with tf.Session() as sess:
        # with tf.device('/gpu:0'):
        writer = tf.summary.FileWriter('./tf_log/', sess.graph)
        # saver = tf.train.Saver()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        data_loader = TextLoader(train_input,batch_size)
        for i in range(epochs):
            data_loader.shuff()
            for j in range(data_loader.num_batches):
                x_train,y_train = data_loader.next_batch(j)
                step, loss_= model.run_step(sess,x_train,y_train)
                saver = tf.train.Saver()
                saver.save(sess, save_path=model_save_path+'model.ckpt', global_step=save_checkpoint_steps)
                print('the epoch number is : %d the index of batch is :%d, the loss value is :%f'%(i, j, loss_))

if mode == 'eval':
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model_path = model_save_path+'model.ckpt-100'
        saver.restore(sess, model_path)
        
        ps,rs = model.evaluate(sess,eval_input)
        ps = np.concatenate((ps),axis=0)
        rs = np.concatenate((rs),axis=0)
        ps = bool_to_value(ps)
        print(ps[-1])
        print(rs[-1])
        # 打印预测文本出来看
        # ps = ps.tolist()
        #rs = rs.tolist()
        #ps = id_to_labeltext(ps)
        #rs = id_to_labeltext(rs)
        print('length of pred is {}'.format(len(ps)))
        print('length of real is {}'.format(len(rs)))
        micro_f1, macro_f1 = f1_np(rs, ps)
        print(micro_f1, macro_f1)


