import tensorflow as tf

input1 = tf.placeholder(tf.float32)  # input1 = tf.placeholder(tf.float32,[2,2]) 可以指定输入格式
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.],input2:[2.]}))