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
def WeightsVariable(shape,name_str,stddev=0.1):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    #initial=tf.truncated_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,name=name_str,dtype=tf.float32)
def BiasesVariable(shape,name_str,stddev=0.00001):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    #initial=tf.constant(stddev,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)
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

#通用的评估函数，用来评估模型在给定的数据集上的损失和准确率(studyai.com)
def EvaluateModelOnDataset(sess, images, labels):
    n_samples = images.shape[0]
    per_batch_size = batch_size
    loss = 0
    acc = 0
    # 样本量比较少的时候，一次性评估完毕；否则拆成若干个批次评估，主要是防止内存不够用
    if (n_samples <= per_batch_size):
        batch_count = 1
        loss, acc = sess.run([cross_entropy_loss, accuracy],
                             feed_dict={X_origin: images,
                                        Y_true: labels,
                                        learning_rate: learning_rate_init})
    else:
        batch_count = int(n_samples / per_batch_size)
        batch_start = 0
        for idx in range(batch_count):
            batch_loss, batch_acc = sess.run([cross_entropy_loss, accuracy],
                                           feed_dict={X_origin: images[batch_start:batch_start + per_batch_size, :],
                                                      Y_true: labels[batch_start:batch_start + per_batch_size, :],
                                                      learning_rate: learning_rate_init})
            batch_start += per_batch_size
            # 累计所有批次上的损失和准确率(studyai.com)
            loss += batch_loss
            acc += batch_acc
    # 返回平均值
    return loss / batch_count, acc / batch_count

#调用函数，构造计算图
with tf.Graph().as_default():
    with tf.variable_scope('mnet'):
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
                weights=WeightsVariable(shape=[5,5,1,conv2d_kernels_num],name_str='weights') #卷积核大小是5*5，一共有16个
                biases=BiasesVariable(shape=[conv2d_kernels_num],name_str='biases')   #一个卷积核要一个bias
                conv_out=Conv2d(X_image,weights,biases,stride=1,padding='VALID')
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
                weights=WeightsVariable(shape=[12*12*conv2d_kernels_num,n_classes],name_str='weights')
                biases=BiasesVariable(shape=[n_classes],name_str='biases')
                YPred_logits=FullyConnected(features,weights,biases,activate=tf.identity,act_name='identity')

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

    print("写入计算图")
    summary_writer=tf.summary.FileWriter(logdir='LogFile/SimpleConvNetForMNIST',graph=tf.get_default_graph())
    summary_writer.close()

    #导入MNIST数据
    mnist=input_data.read_data_sets("../MNIST/",one_hot=True)

    #先用一个list来保存结果，然后一起写入csv文件
    result_list=list()
    #写入参数配置
    result_list.append(['learning_rate',learning_rate_init,
                        'training_epoch',training_epochs,
                        'batch_size',batch_size,
                        'display_step',display_step,
                        'conv1_kernels_num',conv2d_kernels_num])

    #表头
    result_list.append(['train_step', 'train_loss', 'validation_loss',
                         'train_step', 'train_accuracy', 'validation_accuracy'])


    startTime=datetime.datetime.now()

    mnet_var=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='mnet')
    saver=tf.train.Saver(mnet_var)
    #启动计算图
    with tf.Session() as sess:
        sess.run(init)
        total_batches=int(mnist.train.num_examples/batch_size)  #批次总数
        print('batch size=',batch_size)
        print('Train sample count=',mnist.train.num_examples)
        print('batch count=',total_batches)
        training_step=0      #记录模型训练的步数
        #训练指定轮数
        for epoch in range(training_epochs):
            #每一轮要把所有的batch跑一次
            for batch_index in range(int(total_batches/10)):
                #取出数据
                batch_x,batch_y=mnist.train.next_batch(batch_size)
                #运行优化器的训练节点
                sess.run(trainer,feed_dict={X_origin:batch_x,Y_true:batch_y,learning_rate:learning_rate_init})
                #每调用一次节点，step+1
                training_step+=1
                #每训练display_step次，计算当前模型的损失和分类准确率
                if training_step%display_step==0:
                    #计算当前模型在目前见过的display_step个batchsize的训练集上的损失和分类准确率
                    start_index=max(0,(batch_index-display_step)*batch_size)
                    end_index=batch_index*batch_size
                    train_loss,train_acc=EvaluateModelOnDataset(sess,
                                                                mnist.train.images[start_index:end_index],
                                                                mnist.train.labels[start_index:end_index])
                    print("Training Step: " + str(training_step) +
                          ", Training Loss= " + "{:.6f}".format(train_loss) +
                          ", Training Accuracy= " + "{:.5f}".format(train_acc))

                    # 计算当前模型在验证集的损失和分类准确率
                    validation_loss, validation_acc = EvaluateModelOnDataset(sess,
                                                                             mnist.validation.images,
                                                                             mnist.validation.labels)
                    print("Training Step: " + str(training_step) +
                          ", Validation Loss= " + "{:.6f}".format(validation_loss) +
                          ", Validation Accuracy= " + "{:.5f}".format(validation_acc))
                    # 将评估结果保存到文件
                    result_list.append([training_step, train_loss, validation_loss,
                                         training_step, train_acc, validation_acc])


        print("训练完毕!")
        endTime=datetime.datetime.now()
        print("训练耗时：",(endTime-startTime).seconds)


        # 计算指定数量的测试集上的准确率
        test_samples_count = mnist.test.num_examples
        test_loss, test_accuracy = EvaluateModelOnDataset(sess, mnist.test.images, mnist.test.labels)
        print("Testing Samples Count:", test_samples_count)
        print("Testing Loss:", test_loss)
        print("Testing Accuracy:", test_accuracy)

        result_list.append(['test step', 'loss', test_loss, 'accuracy', test_accuracy])

        #保存模型
        saver.save(sess,'ckptfilesScoped/model')
        # 将评估结果保存到文件
        results_file = open('LogFile/csv/evaluate_results.csv', 'w', newline='')
        csv_writer = csv.writer(results_file, dialect='excel')
        for row in result_list:
            csv_writer.writerow(row)















