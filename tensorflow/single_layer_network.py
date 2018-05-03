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


df0 = generate_datablock(15, [7, 7], 22, 0)
df1 = generate_datablock(15, [22, 7], 22, 0)
df2 = generate_datablock(10, [7, 22], 22, 0)
df3 = generate_datablock(25, [20, 20], 22, 1)

df = pd.concat([df0, df1, df2, df3], ignore_index=True)
train_set = df.reindex(permutation(df.index)).reset_index(drop=True)

train_x = train_set[['x1', 'x2']].as_matrix()
train_y = train_set[['y']].as_matrix()


num_units = 4
mult = train_x.flatten().mean()

x = tf.placeholder(tf.float32, [None, 2])
w1 = tf.Variable(tf.truncated_normal([2, num_units]))
b1 = tf.Variable(tf.zeros([num_units]))
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1*mult)

w0 = tf.Variable(tf.zeros([num_units, 1]))
b0 = tf.Variable(tf.zeros([1]))
p = tf.nn.sigmoid(tf.matmul(hidden1, w0) + b0*mult)

y = tf.placeholder(tf.float32, [None, 1])
loss = -tf.reduce_sum(y*tf.log(p) + (1-y)*tf.log(1-p))
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(y-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


i = 0
for _ in range(4000):
    i += 1
    sess.run(train_step, feed_dict={x: train_x, y: train_y})
    if i % 400 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: train_x, y: train_y})
        print('Step: {:5d} / Loss: {:3.3f} / Accuracy: {:3.3f}'.format(i, loss_val, acc_val))


train_set0 = train_set[train_set['y'] == 0]
train_set1 = train_set[train_set['y'] == 1]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.scatter(train_set0.x1, train_set0.x2, marker='x')
ax.scatter(train_set1.x1, train_set1.x2, marker='o')

locations = []
for x1 in np.linspace(0, 30, 100):
    for x2 in np.linspace(0, 30, 100):
        locations.append((x1, x2))

p_vals = sess.run(p, feed_dict={x: locations})
p_vals = p_vals.reshape((100, 100))
ax.imshow(p_vals, origin='lower', extent=(0, 30, 0, 30), cmap='gray_r', alpha=.5)
plt.show()
sess.close()
