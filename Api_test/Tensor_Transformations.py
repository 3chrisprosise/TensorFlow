import tensorflow as tf
import numpy as np

str = tf.string_to_number("2.5555555555555555555555555555555555555555555555555555")  # Transfor string to a float32 or int32 number
double_number = tf.to_double(str)  # Transfro a type to a float64 number
float_number = tf.to_float(double_number) # Transfro a type to a float32 number
bfloat_number = tf.to_bfloat16(float_number)  # Transfro a type to a bfloat16 number
int32_number = tf.to_int32(double_number)   # Transfro a type to a int32 number
int64_number = tf.to_int64(str)   # Transfro a type to a int64 number
cast_number = tf.cast(int64_number, tf.int32) # Transfro a type to another type

t =np.array([[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]])
shape = tf.shape(t)  # return a shape of a tensor of type int32
size = tf.size(t)   # return a tensor size of type int32
rank = tf.rank(t)  # return a tensor rank  of type int32  秩


# Reshape

t = np.array([1,2,3,
              4,5,6,
              7,8,9])
t = tf.reshape(t,[3,3])  # reshape a tensor to a shape that gived as tenser gived
t = tf.reshape(t,[-1])   # flat a tensor into a liner tensor of type as tenser gived

t = np.array([1.0,2.0,3.0],)  # 这边有问题！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
shape = tf.shape(t)   # 删除其中维度为 1 的张量
squeeze = tf.squeeze(shape)
with tf.Session() as Sess:
    print(Sess.run(shape))
