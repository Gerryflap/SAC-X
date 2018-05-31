
import tensorflow as tf
t_indices = [0,1,0,1,1]
t_values = [[1,3], [1,2], [5,6], [0,7], [2,5]]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    selectors = tf.range(tf.shape(t_values)[0], dtype=tf.int32)
    t_indices = tf.stack([selectors, t_indices], axis=1)
    print(sess.run(tf.gather_nd(t_values, t_indices)))