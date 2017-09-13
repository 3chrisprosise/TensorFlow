import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2)  # 矩阵乘法

with tf.Session() as sess:  # Session is an Object so need to be a upercode
    print(sess.run(product))
    # sess.close()  # 关闭session ，with 会自动close 