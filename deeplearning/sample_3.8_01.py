import tensorflow as tf


def my_network(input):
    W_1 = tf.Variable(tf.random_uniform([784, 100], -1, 1))
    b_1 = tf.Variable(tf.zeros([100]))
    output_1 = tf.matmul(input, W_1) + b_1

    W_2 = tf.Variable(tf.random_uniform([100, 50], -1, 1))
    b_2 = tf.Variable(tf.zeros([50]))
    output_2 = tf.matmul(output_1, W_2) + b_2

    W_3 = tf.Variable(tf.random_uniform([50, 10], -1, 1))
    b_3 = tf.Variable(tf.zeros([10]))
    output_3 = tf.matmul(output_2, W_3) + b_3

    print(W_1.name, W_2.name, W_3.name)
    print(b_1.name, b_2.name, b_3.name)

    return output_3
