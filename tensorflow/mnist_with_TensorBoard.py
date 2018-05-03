import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


np.random.seed(20160612)
tf.set_random_seed(20160612)

mnist = input_data.read_data_sets('/data', one_hot=True)


class SingleLayerNetwork:
    def __init__(self, num_units):
        with tf.Graph().as_default():
            self.prepare_model(num_units)
            self.prepare_session()

    def prepare_model(self, num_units):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='input')

        with tf.name_scope('hidden'):
            w1 = tf.Variable(tf.truncated_normal([784, num_units]), name='weights')
            b1 = tf.Variable(tf.zeros([num_units]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='hidden1')

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units, 10]), name='weights')
            b0 = tf.Variable(tf.zeros([10]), name='biases')
            p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0, name='softmax')

        with tf.name_scope('optimizer'):
            y = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = -tf.reduce_sum(y*tf.log(p), name='loss')
            train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(y-0.5))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.histogram('weights_hidden', w1)
        tf.summary.histogram('biases_hidden', b1)
        tf.summary.histogram('weights_output', w0)
        tf.summary.histogram('biases_output', b0)

        self.x, self.y, self.p = x, y, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('/tmp/mnist_sl_logs', sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer


nn = SingleLayerNetwork(1024)

i = 0
for _ in range(2000):
    i += 1
    batch_xs, batch_ys = mnist.train.next_batch(10)
    nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.y: batch_ys})
    if i % 100 == 0:
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss, nn.accuracy],
            feed_dict={nn.x: batch_xs, nn.y: batch_ys})
        print('Step: {:5d} / Loss: {:3.3f} / Accuracy: {:3.3f}'.format(i, loss_val, acc_val))
        nn.writer.add_summary(summary, i)

nn.sess.close()
