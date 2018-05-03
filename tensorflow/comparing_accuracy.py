import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


X, y = make_blobs(n_samples=10000)
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0)
train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

x = tf.placeholder(tf.float32, [None, 2])
w = tf.Variable(tf.zeros([2, 1]))
w0 = tf.Variable(tf.zeros([1]))
f = tf.matmul(x, w) + w0
p = tf.sigmoid(f)

y = tf.placeholder(tf.float32, [None, 1])
loss = -tf.reduce_sum(y*tf.log(p) + (1-y)*tf.log(1-p))
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(y-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


train_accuracy = []
test_accuracy = []

for _ in range(2500):
    sess.run(train_step, feed_dict={x: train_x, y: train_y})
    train_accuracy.append(sess.run(accuracy, feed_dict={x: train_x, y: train_y}))
    test_accuracy.append(sess.run(accuracy, feed_dict={x: test_x, y: test_y}))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(len(train_accuracy)), train_accuracy, label='Training Accuracy')
ax.plot(range(len(test_accuracy)), test_accuracy, label='Test Accuracy', linestyle='--')
ax.legend(loc='best')
plt.show()

sess.close()
