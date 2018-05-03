import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 変数の定義
x = tf.placeholder(tf.float32, [None, 5])
w = tf.Variable(tf.zeros([5, 1]))
y = tf.matmul(x, w)
t = tf.placeholder(tf.float32, [None, 1])


# コスト関数とオプティマイザを定義
loss = tf.reduce_sum(tf.square(y - t))
train_step = tf.train.AdamOptimizer().minimize(loss)


# 初期化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


# 訓練データの作成（今回はお手製）
train_t = np.array([5.2, 5.7, 8.6, 14.9, 18.2, 20.4,
                    25.5, 26.4, 22.8, 17.5, 11.1, 6.6])
train_t = train_t.reshape(12, 1)
train_x = np.zeros([12, 5])

for row, month in enumerate(range(1, 13)):
    for col, n in enumerate(range(0, 5)):
        train_x[row][col] = month**n


# 学習開始
i = 0
for _ in range(100000):
    i += 1
    _, loss_val = sess.run([train_step, loss], feed_dict={x: train_x, t: train_t})
    if i % 10000 == 0:
        # loss_val = sess.run(loss, feed_dict={x: train_x, t: train_t})
        print('Step: {:7d} | Loss: {:3.6f}'.format(i, loss_val))


# 学習後のパラメータを表示
w_val = sess.run(w)
print('weight:\n{}'.format(w_val))


# 予測式
def predict(x):
    result = 0.0
    for n in range(0, 5):
        result += w_val[n][0] * x**n
    return result


# グラフの描画
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

line = np.linspace(1, 12, 100)
ax.scatter(range(1, 13), train_t)
ax.plot(line, predict(line))

ax.set_xlabel('Month')
ax.set_ylabel('Average Temperature')
ax.set_title('Monthly Average Temperature')
ax.set_xlim(1, 12)
plt.show()
