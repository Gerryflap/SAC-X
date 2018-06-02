"""
    This file is used to test the implementation of the loss functions
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


policy_raw = tf.Variable([[0.0, 0.0]])
policy = tf.keras.activations.softmax(policy_raw)
values_t = tf.constant([[0.0, 1.0]])

entropy_regularization = tf.placeholder(tf.float32)

# This is the only working configuration:
task_policy_score = -tf.reduce_sum(policy * (values_t - entropy_regularization * tf.log(policy)))

optimizer = tf.train.AdamOptimizer(0.1)
gradients = tf.gradients(task_policy_score, policy_raw)[0]
optimize = optimizer.apply_gradients([(gradients, policy_raw)])

plots = []
for entropy_regularization_v, color in zip([0.1, 1.0, 10], ['red', 'green', 'blue']):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        policy_values = np.zeros((100, 2))
        for i in range(100):
            policy_v, gradients_v, _ = sess.run([policy, gradients, optimize], feed_dict={entropy_regularization: entropy_regularization_v})
            policy_values[i] = policy_v
            print(policy_v, gradients_v)
        plots.append(plt.plot(policy_values[:, 1], color=color)[0])
plt.legend(plots, [0.1, 1.0, 10], title="Entropy reg. factor")
plt.title("Learned probability for the optimal action in a binary action space")
plt.xlabel("Time (steps)")
plt.ylabel("P(a=1)")
plt.show()