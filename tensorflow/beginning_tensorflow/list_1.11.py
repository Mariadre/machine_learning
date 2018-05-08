import tensorflow as tf

const1 = tf.constant(2, name='num1')
const2 = tf.constant(3, name='num2')
add_op = tf.add(const1, const2, name='sum')
mul_op = tf.multiply(add_op, const2, name='times')

with tf.Session() as sess:
    mul_result, add_result = sess.run([mul_op, add_op])
    print(mul_result)
    print(add_result)

    tf.summary.FileWriter('./chkpnt', sess.graph)
    sess.close()
