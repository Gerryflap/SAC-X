import tensorflow as tf

from SAC_U.sac_u import SacU

ks = tf.keras
import environments.mock_env as mock_env
import gym
import numpy as np


def policy_model2(t_id, x):
    x = ks.layers.Dense(1, activation='elu')(x)
    xs = []
    for i in range(3):
        nx = ks.layers.Dense(3, activation='elu')(x)
        xs.append(ks.layers.Dense(3, activation='softmax')(nx))
    xs = tf.stack(xs, axis=1)
    batch_indices = tf.range(tf.shape(t_id)[0])
    selectors = tf.stack([batch_indices, t_id], axis=1)
    return tf.gather_nd(xs, selectors)


def value_model2(t_id, action, x):
    #x = tf.concat([x, action], axis=1)
    x = ks.layers.Dense(10, activation='elu')(x)
    xs = []
    for i in range(3):
        nx = ks.layers.Dense(10, activation='elu')(x)
        xs.append(ks.layers.Dense(3, activation='linear')(nx))
    xs = tf.stack(xs, axis=1)
    batch_indices = tf.range(tf.shape(t_id)[0])
    selectors = tf.stack([batch_indices, t_id], axis=1)
    return tf.reduce_sum(tf.gather_nd(xs, selectors)*action, axis=1, keepdims=True)

def simple_policy_model(t_id, state):
    weights = tf.Variable(tf.random_normal((1,3)), name="Policy_weights")
    #constants = tf.constant([[1, 1, 1]], dtype=tf.float32) + 0 * weights
    return tf.tile(ks.activations.softmax(weights, axis=1), [tf.shape(state)[0], 1])

def simple_value_model(t_id, action, x):
    weights = tf.Variable(tf.random_normal((1, 3)), name="Value_weights")
    #constants = tf.constant([[1,0,0]], dtype=tf.float32) + 0*weights
    return tf.reduce_sum(weights*action, keepdims=True ,axis=1)



env = lambda: mock_env.MockEnv()
sac_u = SacU(policy_model2, value_model2, env, (3,), mock_env.action_space, 1, 1, buffer_size=100, visual=False, averaged_gradients=1,
             learning_rate=0.01, entropy_regularization_factor=0.5, scheduler_period=200, max_steps=1000, gamma=0.5)
sac_u.run()