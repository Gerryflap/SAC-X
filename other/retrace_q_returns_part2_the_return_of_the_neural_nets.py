"""
    In this file the retrace algorithm is tested (the basic Q returns without the c_k)
"""

import random
from collections import defaultdict
import tensorflow as tf
import numpy as np

ks = tf.keras

gamma = 0.5

# c_k is omitted for simplicity: same policy so always 1
action_space = [0, 1]


def value_model(s):
    x = ks.layers.Dense(100, activation='elu', input_shape=(2,))(s)
    return ks.layers.Dense(2, activation='linear')(x)


state = tf.placeholder(tf.float32, (None, 2))
with tf.variable_scope("current"):
    q = value_model(state)
with tf.variable_scope("fixed"):
    q_fixed = value_model(state)
action_t = tf.placeholder(tf.int32)
q_ret_t = tf.placeholder(tf.float32, (None, 1))
loss_t = tf.reduce_sum(tf.square(q*tf.one_hot(action_t, 2)-q_ret_t))
optimize_op = tf.train.AdamOptimizer(0.01).minimize(loss_t)

def policy_calc(q_values):
    if len(q_values.shape) == 1:
        policy = np.zeros((2,)) + 0.2
        policy[np.argmax(q_values)] = 0.8
    else:
        policy = np.zeros((q_values.shape[0],2)) + 0.2
        #selections = np.stack((np.arange(q_values.shape[0]), np.argmax(q_values, axis=1)), axis=1)
        selections = np.argmax(q_values, axis=1)
        for j in range(q_values.shape[0]):
            policy[j, selections[j]] = 0.8
    return policy


# def get_q_return(trajectory):
#     expected_q_s0 = sum([policy[a] * q_values[(0, a)] for a in action_space])
#     q_return = 0
#     for j, (s, a) in enumerate(trajectory[:-1]):
#         q_delta = expected_q_s0 - q_values[(s,a)]
#         q_return += rewards[(s, a)] + q_delta
#     return q_return

def get_q_delta(trajectory, sess):
    q_delta = 0
    cum_reward = 0
    q_values = sess.run(q_fixed, feed_dict={state:[e[0] for e in trajectory]})
    policies = policy_calc(q_values)
    for j, (s, a, r) in enumerate(trajectory):
        #cum_reward += gamma**j * r
        if j != len(trajectory) - 1:
            expected_q_tp1 = np.sum(policies[j + 1] * q_values[j + 1])
            delta = r + gamma * expected_q_tp1 - q_values[j, a]
        else:
            delta = r - q_values[j, a]
        q_delta += gamma**j * delta

    return q_delta

def copy_vars(sess):
    vars = tf.trainable_variables("current")
    vals = sess.run(vars)
    var_dict = dict()
    for var, val in zip(vars, vals):
        var_dict[var.name.replace("current/", "")] = val

    for var in tf.trainable_variables("fixed"):
        name = var.name.replace("fixed/", "")
        if name in var_dict:
            tf.assign(var, var_dict[name]).eval()


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(100000):
        trajectory = []
        for j in range(100):
            s = [j/100.0,0]
            q_vals = sess.run(q, feed_dict={state: [s]})[0]
            action = np.random.choice([0,1], p=policy_calc(q_vals))
            r = action
            trajectory.append((s, action, r))
        trajectory.append(([0,1], 0, 0))

        qrets = []
        old_qs = sess.run(q_fixed, feed_dict={state: [e[0] for e in trajectory]})
        for j in range(len(trajectory)):
            qrets.append([get_q_delta(trajectory[j:], sess) + old_qs[j][trajectory[j][1]]])
        #print(qrets[:5], [e[1] for e in trajectory][:5])
        #q_return = get_q_return(trajectory)
        #q_values[(s,a)] -= alpha*2*(q_values[(s,a)] - q_return)
        sess.run(optimize_op, feed_dict={state: [e[0] for e in trajectory], action_t:[e[1] for e in trajectory], q_ret_t: qrets})
        print(sess.run(q, feed_dict={state: [trajectory[0][0]]}))
        print(sess.run(loss_t, feed_dict={state: [e[0] for e in trajectory], action_t:[e[1] for e in trajectory], q_ret_t: qrets}))
        if i%100 == 0:
            copy_vars(sess)