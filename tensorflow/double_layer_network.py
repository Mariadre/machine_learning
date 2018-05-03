import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
from pandas import DataFrame


np.random.seed(20160614)
tf.set_random_seed(20160614)


def generate_datablock(m, mu, var, y):
    data = multivariate_normal(mu, np.eye(2)*var, m)
    df = DataFrame(data, columns=['x1', 'x2'])
    df['y'] = y
    return df


df0 = generate_datablock(30, [-7, -7], 18, 1)
df1 = generate_datablock(30, [-7, 7], 18, 0)
df2 = generate_datablock(30, [7, -7], 18, 0)
df3 = generate_datablock(30, [7, 7], 18, 1)

df = pd.concat([df0, df1, df2, df3], ignore_index=True)
train_set = df.reindex(permutation(df.index)).reset_index(drop=True)

train_x = train_set[['x1', 'x2']].as_matrix()
train_y = train_set[['y']].as_matrix()


num_units1 = 2
num_units2 = 2

x = tf.placeholder(tf.float32, [None, 2])
w1 = tf.Variable(tf.truncated_normal([2, num_units1]))
b1 = tf.Variable(tf.zeros([num_units1]))
hidden1 = tf.nn.tanh(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.tanh(tf.matmul(hidden1, w2) + b2)

w0 = tf.Variable(tf.zeros([num_units2, 1]))
b0 = tf.Variable(tf.zeros([1]))
p = tf.nn.sigmoid(tf.matmul(hidden2, w0) + b0)


y = tf.placeholder(tf.float32, [None, 1])
loss = -tf.reduce_sum(y*tf.log(p) + (1-y)*tf.log(1-p))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(y-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


i = 0
for _ in range(1000):
    i += 1
    sess.run(train_step, feed_dict={x: train_x, y: train_y})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: train_x, y: train_y})
        print('Step: {:5d} / Loss: {:3.3f} / Accuracy: {:3.3f}'.format(i, loss_val, acc_val))


train_set0 = train_set[train_set['y'] == 0]
train_set1 = train_set[train_set['y'] == 1]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.scatter(train_set0.x1, train_set0.x2, marker='x')
ax.scatter(train_set1.x1, train_set1.x2, marker='o')

locations = []
for x1 in np.linspace(-15, 15, 100):
    for x2 in np.linspace(-15, 15, 100):
        locations.append((x1, x2))

p_vals = sess.run(p, feed_dict={x: locations})
p_vals = p_vals.reshape((100, 100))
ax.imshow(p_vals, origin='lower', extent=(-15, 15, -15, 15), cmap='gray_r', alpha=.5)
plt.show()

sess.close()
