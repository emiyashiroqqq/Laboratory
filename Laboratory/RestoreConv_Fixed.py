import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import csv
import datetime

#算法超参数
learning_rate_init=0.001
training_epochs=1
batch_size=100
display_step=10

#网络参数
n_input=784            #MNIST输入为28*28=784
n_classes=10           #分为10类

conv2d_kernels_num=64   #卷积核数量

#生成权重节点和偏置节点的函数
def WeightsVariable(shape,name_str,stddev=0.1,is_train=True):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    #initial=tf.truncated_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,name=name_str,dtype=tf.float32,trainable=is_train)
def BiasesVariable(shape,name_str,stddev=0.00001,is_train=True):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    #initial=tf.constant(stddev,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name=name_str,trainable=is_train)
#二维卷积操作函数
def Conv2d(x,W,b,stride=1,padding='SAME'):
    with tf.name_scope('Wx_plus_b'):
        y=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
        y=tf.nn.bias_add(y,b)
    return y

#非线性激活层
def Activation(x,activation=tf.nn.relu,name='relu'):
    with tf.name_scope(name):
        y=activation(x)
    return y

#二维池化操作
def Pool2d(x,pool=tf.nn.max_pool,k=2,stride=2):
    return pool(x,ksize=[1,k,k,1],strides=[1,stride,stride,1],padding='VALID')

#全连接层，activate为identity就是线性的，为其他非线性激活函数就是非线性
def FullyConnected(x,W,b,activate=tf.nn.relu,act_name='relu'):
    with tf.name_scope('Wx_plus_b'):
        y=tf.matmul(x,W)
        y=tf.add(y,b)
    with tf.name_scope(act_name):
        y=activate(y)
    return y


#调用函数，构造计算图
with tf.Graph().as_default():
    #输入
    with tf.name_scope('Inputs'):
        X_origin=tf.placeholder(tf.float32,[None,n_input],name='X_origin')
        Y_true=tf.placeholder(tf.float32,[None,n_classes],name='Y_true')
        #将图像从N*784的张量转换为N*28*28*1的张量
        X_image=tf.reshape(X_origin,[-1,28,28,1])
    #开始前向推断
    with tf.name_scope('Inference'):
        #第一个卷积层(conv2d+biases)
        with tf.name_scope("Conv2d"):
            weights1=WeightsVariable(shape=[5,5,1,conv2d_kernels_num],name_str='weights',is_train=False) #卷积核大小是5*5，一共有16个
            biases1=BiasesVariable(shape=[conv2d_kernels_num],name_str='biases',is_train=False)   #一个卷积核要一个bias
            conv_out=Conv2d(X_image,weights1,biases1,stride=1,padding='VALID')
        #非线性激活层
        with tf.name_scope('Activate'):
            activate_out=Activation(conv_out,activation=tf.nn.relu,name='relu')
        #池化层
        with tf.name_scope('Pool2d'):
            pool_out=Pool2d(activate_out,tf.nn.max_pool,k=2,stride=2)
        #线性全连接
        with tf.name_scope('FeatsReshape'):
            features=tf.reshape(pool_out,[-1,12*12*conv2d_kernels_num])
        with tf.name_scope('FC_Linear'):
            weights2=WeightsVariable(shape=[12*12*conv2d_kernels_num,n_classes],name_str='weights',is_train=False)
            biases2=BiasesVariable(shape=[n_classes],name_str='biases',is_train=True)
            YPred_logits=FullyConnected(features,weights2,biases2,activate=tf.identity,act_name='identity')

        with tf.name_scope('AddTestLayer'):
            weights = WeightsVariable(shape=[n_classes, n_classes], name_str='weights',
                                      is_train=True)
            biases = BiasesVariable(shape=[n_classes], name_str='biases', is_train=False)
            ty=FullyConnected(YPred_logits,weights,biases,activate=tf.identity,act_name='identity')
    #损失层
    with tf.name_scope('Loss'):
        cross_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=Y_true,logits=YPred_logits))

    #定义优化训练层
    with tf.name_scope('Train'):
        learning_rate=tf.placeholder(tf.float32)
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        trainer=optimizer.minimize(cross_entropy_loss)
    #定义模型评估层
    with tf.name_scope('Evaluate'):
        correct_pred=tf.equal(tf.argmax(YPred_logits,1),tf.argmax(Y_true,1))
        accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='res')


    #初始化节点
    init=tf.global_variables_initializer()

    saver = tf.train.Saver([weights1, weights2, biases1, biases2])
    #Session在如CKPT文件
    with tf.Session() as sess:
        #载入模型参数值
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint('ckptfiles/'))
        mnist = input_data.read_data_sets("../MNIST/", one_hot=True)
        image = mnist.test.images
        label = mnist.test.labels
        acc = sess.run(accuracy, feed_dict={X_origin: image[:100], Y_true: label[:100]})
        print("载入后的正确率",acc)
        #再训练
        # 训练指定batch_num和batch_size
        batch_num=30
        batch_size=100
        for epoch in range(batch_num):
            # 取出数据
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 运行优化器的训练节点
            sess.run(trainer, feed_dict={X_origin: batch_x, Y_true: batch_y, learning_rate: learning_rate_init})
        print("训练完毕!")
        acc = sess.run(accuracy, feed_dict={X_origin: image[:100], Y_true: label[:100]})
        print("再训练后的正确率",acc)