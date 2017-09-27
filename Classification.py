import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MINIST_data',one_hot=True)

def add_layer(inputs, insize, outsize,activation_function):

        with tf.name_scope('weights'):
            Weights= tf.Variable(tf.random_normal([insize,outsize]))
        with tf.name_scope('biases'):
            biasis = tf.Variable(tf.zeros([1, outsize]) + 0.1)  #
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biasis
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


xs = tf.placeholder(tf.float32, [None, 784]) #28 *28
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_mean(ys*tf.log(prediction),reduction_indices=[1]))
#softmax 和cross_entropy 算法结合处理分类问题
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)





with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    def compute_accuracy(Veridation_xs, Veridation_ys):
        global prediction
        y_pre = sess.run(prediction,feed_dict={xs:Veridation_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(Veridation_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        result = sess.run(accuracy, feed_dict={xs:Veridation_xs, ys:Veridation_ys})
        return result
    for i in range(10000):
        merged = tf.summary.merge_all()
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
           print(compute_accuracy(mnist.test.images, mnist.test.labels))
#  正规化
#L1 ,L2 ,....regulation
# Y = WX
# COST = (WX  - real_y)^2 + abs(w)