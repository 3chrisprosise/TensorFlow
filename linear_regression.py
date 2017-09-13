# 线性回归

#  创建网络结构
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

Weight = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))  # tf.zeros -> tf.zeros

y = Weight * x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))  # 计算预测值和真实值差值
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 梯度下降参数

train = optimizer.minimize(loss)
# 网络结构创建结束
# 初始化网络 结束

init = tf.initialize_all_variables()  # 激活 变量

sess = tf.Session()

sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weight),sess.run(biases))
