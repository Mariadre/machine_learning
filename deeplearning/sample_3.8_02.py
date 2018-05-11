import tensorflow as tf


def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', weight_shape, initializer=weight_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    print(W.name, b.name)
    return tf.matmul(input, W) + b


def my_network(input):
    with tf.variable_scope('layer-1'):
        output_1 = layer(input, [784, 100], [100])

    with tf.variable_scope('layer-2'):
        output_2 = layer(output_1, [100, 50], [50])

    with tf.variable_scope('layer-3'):
        output_3 = layer(output_2, [50, 10], [10])

    return output_3


if __name__ == '__main__':
    with tf.variable_scope('shared_variables') as scope:
        i_1 = tf.placeholder(tf.float32, [1000, 784], name='i_1')
        print(my_network(i_1))
        scope.reuse_variables()  # allow reusing variables
        i_2 = tf.placeholder(tf.float32, [1000, 784], name='i_2')
        print(my_network(i_2))
