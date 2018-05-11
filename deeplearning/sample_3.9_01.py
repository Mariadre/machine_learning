import tensorflow as tf


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name='a')
        b = tf.constant([1.0, 2.0], shape=[2, 1], name='b')
        c = tf.matmul(a, b)

        x = tf.constant(3)
        y = tf.constant(4)
        z = tf.add_n([x, y, x])

    # log_device_placement --> 各グラフノードの計算に用いられたデバイスログが出力される
    # allow_soft_placement --> 指定デバイスが存在しない場合（e.g. '/gpu:999'）でも代替デバイスで処理を続行する
    config = tf.ConfigProto(log_device_placement=True,
                            allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        sess.run(c)
        sess.run(z)
