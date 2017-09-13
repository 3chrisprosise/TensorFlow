import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)

# new_value = tf.assign(state, one)
update = tf.initialize_all_variables()

with tf.Session() as sess:
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
