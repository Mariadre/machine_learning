import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
from pandas import DataFrame

np.random.seed(20160512)

# negative
m0, mu0, variance0 = 20, [10, 11], 20
data0 = multivariate_normal(mu0, np.eye(2)*variance0, m0)
df0 = DataFrame(data0, columns=['x1', 'x2'])
df0['y'] = 0

# positive
m1, mu1, variance1 = 15, [18, 20], 22
data1 = multivariate_normal(mu1, np.eye(2)*variance1, m1)
df1 = DataFrame(data1, columns=['x1', 'x2'])
df1['y'] = 1

df = pd.concat([df0, df1], ignore_index=True)
train_set = df.reindex(permutation(df.index)).reset_index(drop=True)


train_x = train_set[['x1', 'x2']].as_matrix()
train_y = train_set[['y']].as_matrix()


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


for i, _ in enumerate(range(20001)):
    _, loss_val, acc_val = sess.run([train_step, loss, accuracy], feed_dict={x: train_x, y: train_y})
    if i % 2000 == 0:
        print('Step: {:6d} / Loss: {:.3f} / Accuracy: {:.3f}'.format(i, loss_val, acc_val))

w0_val, w_val = sess.run([w0, w])
print('intercept: {}'.format(w0_val))
print('coefficient: {}'.format(w_val))


train_set0 = train_set[train_set['y'] == 0]
train_set1 = train_set[train_set['y'] == 1]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(np.min(train_set.x1), np.max(train_set.x1))
ax.set_ylim(np.min(train_set.x2), np.max(train_set.x2))
ax.scatter(train_set0.x1, train_set0.x2, marker='x')
ax.scatter(train_set1.x1, train_set1.x2, marker='o')

line_x = np.linspace(0, 30, 100)
line_y = -((w_val[0][0]*line_x) + w0_val) / w_val[1][0]
ax.plot(line_x, line_y)

plt.show()
sess.close()
