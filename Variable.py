import tensorflow as tf

state = tf.Variable(3.0 ,name='counter')  # in tf variable is an Object so need to be upper code
print(state.name)  # a variable has a name
one = tf.constant(1.0)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)   # 将第二个参数赋值给第一个参数

init = tf.initialize_all_variables()  # 定义变量一定要初始化

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))