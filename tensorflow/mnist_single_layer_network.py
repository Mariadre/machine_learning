import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


np.random.seed(20160612)
tf.set_random_seed(20160612)

mnist = input_data.read_data_sets('/data', one_hot=True)


num_units = 1024

x = tf.placeholder(tf.float32, [None, 784])
w1 = tf.Variable(tf.truncated_normal([784, num_units]))
b1 = tf.Variable(tf.zeros([num_units]))
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w0 = tf.Variable(tf.zeros([num_units, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)

y = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(y*tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(y-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


for i, _ in enumerate(range(2000)):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
                                     feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print('Step: {:5d} / Loss: {:3.3f} / Accuracy: {:3.3f}'.format(i, loss_val, acc_val))


images, labels = mnist.test.images, mnist.test.labels
p_val = sess.run(p, feed_dict={x: images, y: labels})

fig = plt.figure(figsize=(8, 15))
for i in range(100):
    c = 1
    for (image, label, pred) in zip(images, labels, p_val):
        prediction, actual = np.argmax(pred), np.argmax(label)
        if prediction != i:
            continue
        if (c < 4 and i == actual) or (c >= 4 and i != actual):
            ax = fig.add_subplot(10, 6, i*6+c)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('{} / {}'.format(prediction, actual))
            ax.imshow(image.reshape(28, 28), cmap='gray_r', interpolation='nearest')

            c += 1
            if c > 6:
                break

plt.show()
