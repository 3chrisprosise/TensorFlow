# 少量层中随意尝试激励函数
# 多层推荐relu    否则可能产生梯度爆炸
# 循环网络为relu or tanh

import tensorflow as tf
import numpy as np
# 定义神经层
def add_layer(inputs, insize, outsize,activation_function):
    Weight = tf.Variable(tf.random_normal([insize,outsize]))
    biasis = tf.Variable(tf.zeros([1,outsize]) + 0.1) #
    Wx_plus_b = tf.matmul(inputs,Weight)+ biasis
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 制造源数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
nosie = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + nosie  # 求平方

# 构造隐含层
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
lay1 = add_layer(xs,1, 10, activation_function=tf.nn.relu)
prediction = add_layer(lay1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                      reduction_indices=[1]))# 检测方差？ reduce_mean  求平均值

# 选择训练函数  梯度下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))