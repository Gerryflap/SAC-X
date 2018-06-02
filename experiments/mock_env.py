import tensorflow as tf

from SAC_U.sac_u import SacU

ks = tf.keras
import environments.mock_env as mock_env
import gym
import numpy as np


def policy_model2(t_id, x):
    x = ks.layers.Dense(100, activation='elu')(x)
    xs = []
    for i in range(2):
        nx = ks.layers.Dense(100, activation='elu')(x)
        xs.append(ks.layers.Dense(3, activation='softmax')(nx))
    xs = tf.stack(xs, axis=1)
    batch_indices = tf.range(tf.shape(t_id)[0])
    selectors = tf.stack([batch_indices, t_id], axis=1)
    return tf.gather_nd(xs, selectors)


def value_model2(t_id, action, x):
    x = tf.concat([x, action], axis=1)
    x = ks.layers.Dense(100, activation='elu')(x)
    xs = []
    for i in range(2):
        nx = ks.layers.Dense(100, activation='elu')(x)
        xs.append(ks.layers.Dense(1, activation='linear')(nx))
    xs = tf.stack(xs, axis=1)
    batch_indices = tf.range(tf.shape(t_id)[0])
    selectors = tf.stack([batch_indices, t_id], axis=1)
    return tf.gather_nd(xs, selectors)


env = lambda: mock_env.MockEnv()
sac_u = SacU(policy_model2, value_model2, env, (3,), mock_env.action_space, 1, 2, buffer_size=10000, visual=False, averaged_gradients=1,
             learning_rate=0.003, entropy_regularization_factor=5.0, scheduler_period=200)
sac_u.run()