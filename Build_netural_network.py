# 少量层中随意尝试激励函数
# 多层推荐relu    否则可能产生梯度爆炸
# 循环网络为relu or tanh

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 定义神经层
def add_layer(inputs, insize, outsize, n_layer,activation_function):
    layer_name = 'layers%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights= tf.Variable(tf.random_normal([insize,outsize]))
            tf.summary.histogram(layer_name+'/weights', Weights) # 总结过程
        with tf.name_scope('biases'):
            biasis = tf.Variable(tf.zeros([1, outsize]) + 0.1)  #
            tf.summary.histogram(layer_name+'/biases', biasis)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biasis
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/output', outputs)
        return outputs


# 制造源数据

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
nosie = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + nosie  # 求平方

# 构造隐含层
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

lay1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(lay1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                          reduction_indices=[1]))# 检测方差？ reduce_mean  求平均值
    tf.summary.scalar('loss',loss)  # 纯量的变化

with tf.name_scope('train'):
# 选择训练函数  梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    merged = tf.summary.merge_all()  #  合并所有的总结图像
    writer = tf.summary.FileWriter('sum', sess.graph)
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()  # show 的时候不暂停
    plt.show()
    for i in range(10000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss,
                           feed_dict={xs: x_data, ys: y_data}))
            result = sess.run(merged,
                              feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs:x_data})
            lines = ax.plot(x_data,prediction_value, 'r-', lw=5)
            plt.pause(0.01)

#  在tensorboard中显示内容 tensorboard --logdir=''  引号中填写生成的board文件所在的文件夹,写路径要用 \\  单个表示转义