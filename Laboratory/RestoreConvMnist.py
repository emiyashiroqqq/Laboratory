import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import csv
import datetime

#加载网络图模型（不包括参数值）
saver=tf.train.import_meta_graph('ckptfiles/model.meta')
graph = tf.get_default_graph()
data = graph.get_tensor_by_name('Inputs/X_origin:0')
label_y = graph.get_tensor_by_name('Inputs/Y_true:0')
acc = graph.get_tensor_by_name('Evaluate/res:0')
#下面的代码可以检查网络结构
# tvs = [v for v in tf.trainable_variables()]
# print("可训练变量")
# for v in tvs:
#     print(v.name)
# gv = [v for v in tf.global_variables()]
# print("global变量")
# for v in gv:
#     print(v.name)
# ops = [o for o in sess.graph.get_operations()]
# print("op变量")
# for o in ops:
#     print(o.name)
#
with tf.Session() as sess:
    #载入模型参数值
    saver.restore(sess, tf.train.latest_checkpoint('ckptfiles/'))
    # 导入MNIST数据
    mnist = input_data.read_data_sets("../MNIST/", one_hot=True)
    image=mnist.test.images
    label=mnist.test.labels
    accuracy=sess.run(acc,feed_dict={data:image[:100],label_y:label[:100]})
    print(accuracy)