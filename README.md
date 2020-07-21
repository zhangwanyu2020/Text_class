# Text_class

文本分类


bayes_demo


1、从每个分类文件夹分别取10、20、30、40条数据进行训练，共计800条数据

2、800条训练，词表维度上升到11000+，训练数据增大到2+万，onehot向量维度维度可以增大7+万，训练不动

3、800条训练，取20条测试数据，测试精度为70%。其中一级分类中，有1个是错的，二级分类中有6个错的。如果增大训练集精确度还可以上升


fasttext


1、每个样本的标签熟练为6到10多个不等，标签集合为95个，只截取前三个，当做多分类问题处理

2、采用2万多条数据训练，测试精度为0.7

3、训练速度较快


textcnn


1、全量数据2.2万+，80%做训练，20%测试

2、词向量不做预训练，测试结果micro_f1=0.407，macro_f1=0.01，二者相差较大

3、训练速度很快。预训练词向量word2vec，测试结果无显著提升


bert


1、bert预训练模型+一层全连接，输出是1*95维的概率分布

2、当只截取前三个层级的标签时，分类的准确率达0.9

3、当纳入全部标签时，确定一个概率阀值判定每个标签是正还是负，阀值的确定考虑的了两种方式：一是确定k个概率最大的值，将其作为正类，这样每个batch的阀值是动态的；二是以一个固定的概率值，所有样本通用，比如0.8。考虑到每个标签的标签数量不是固定的，所以选用后者。

4 、测试结果：f1_score并没有太大提升，micro_f1=0.125，macro_f1=0.1

5、考虑f1_score没有提升的原因，由于损失函数是正常下降的，所以初步判定问题是出在预测的时候，所以打印输出的概率值来看，发现分布太均匀，概率阀值很难将正负两类分开。

6、解决办法：

    1>尝试调整参数、修改损失函数（取对数、取平均、概率值代替预测值...）、加全连接层、甚至修改sigmoid函数，都不好使
    
    2>用tanh函数代替sigmoid,将样本标签0 ~ 1 替换成-1 ~ +1，发现输出的分布出现两极分化，并且少数正样本是被命中的，但是f1_score还是没有提升
    
    3>从tanh的试验中发现一个问题，如果：
    
    pred = [1 1 1 1 0 1 0 0 0 1]
    
    real = [0 0 0 0 1 0 1 0 0 0]
    
    loss = 8
    
    pred2 =[0 0 0 0 0 0 0 0 0 1]
    
    loss = 3
    
    从pred到pred2loss虽然下降了5个点，但是loss下降都体现在负样本上，对于我们想要的正样本是没有意义的。

考虑了两种方式来提升对正样本的准确率：

一是给正样本和负样本的损失给予不同的权重，比如0.8*square(正样本预测值-正样本实际值)+0.2*square(负样本预测值-负样本实际值)；

二是直接给样本的损失乘一个倍数，比如square(正样本预测值-正样本实际值*3)，因为label是0-1向量，只会使正样本的损失加倍。

第一种方式在tf 1.x实现稍微复杂，在tf 2.x很简单；第二种方式操作就更简单了，所以首选第二种。

7、测试结果：f1_score提升很大，micro_f1=0.737，macro_f1=0.314，相比于textcnn几乎翻了一倍
    
    
  
    
    






