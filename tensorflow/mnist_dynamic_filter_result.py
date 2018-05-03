import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


np.random.seed(20160703)
tf.set_random_seed(20160703)

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)


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
saver.restore(sess, '/tmp/chkpnt')


filter_vals, conv_vals, pool_vals = sess.run(
    [w_conv, h_conv, h_pool], feed_dict={x: mnist.test.images[:9]}
)

fig = plt.figure(figsize=(10, num_filters+1))

for i in range(num_filters):
    ax = fig.add_subplot(num_filters+1, 10, 10*(i+1)+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(filter_vals[:, :, 0, i],
              cmap='gray_r', interpolation='nearest')

for i in range(9):
    ax = fig.add_subplot(num_filters+1, 10, i+2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('{}'.format(np.argmax(mnist.test.labels[i])))
    ax.imshow(mnist.test.images[i].reshape((28, 28)),
              vmin=0, vmax=1,
              cmap='gray_r', interpolation='nearest')

    for f in range(num_filters):
        ax = fig.add_subplot(num_filters+1, 10, 10*(f+1)+i+2)
        ax.set_xticks([])
        ax.set_yticks([])
        # conv_vals <--> pool_vals
        ax.imshow(conv_vals[i, :, :, f],
                  cmap='gray_r', interpolation='nearest')


fig = plt.figure(figsize=(12, 10))
c = 0
for (image, label) in zip(mnist.test.images, mnist.test.labels):
    p_val = sess.run(p, feed_dict={x: [image]})
    pred = p_val[0]
    prediction, actual = np.argmax(pred), np.argmax(label)
    if prediction == actual:
        continue
    ax = fig.add_subplot(5, 4, c*2+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('{} / {}'.format(prediction, actual))
    ax.imshow(image.reshape((28, 28)), vmin=0, vmax=1,
              cmap='gray_r', interpolation='nearest')
    ax = fig.add_subplot(5, 4, c*2+2)
    ax.set_xticks(range(10))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(0.1)
    ax.bar(range(10), pred, align='center')
    c += 1
    if c == 10:
        break

plt.show()
sess.close()
