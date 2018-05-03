import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


np.random.seed(20160703)
tf.set_random_seed(20160703)

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)


# settings
num_filters = 16
num_units1 = 14*14*num_filters
num_units2 = 1024


# input layer
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# hidden layer -1
w_conv = tf.Variable(tf.truncated_normal([5, 5, 1, num_filters], stddev=0.1))
h_conv = tf.nn.conv2d(x_image, w_conv, strides=[1, 1, 1, 1], padding='SAME')
h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
h_pool_flat = tf.reshape(h_pool, [-1, 14*14*num_filters])

# hidden layer -2
w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)

# output layer
w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)


# optimizer
y = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(y*tf.log(p))
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

# evaluator
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
saver = tf.train.Saver()


# learning
i = 0
for _ in range(4000):
    i += 1
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i % 400 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: batch_xs, y: batch_ys})
        print('Step: {:5d} / Loss: {} / Accuracy: {}'.format(i, loss_val, acc_val))
        saver.save(sess, '/tmp/chkpnt/cnn', global_step=i)


sess.close()
