import tensorflow as tf

from SAC_U.sac_u import SacU

ks = tf.keras
import environments.wrapper_env as wenv
import gym
import numpy as np
import environments.mountaincar_long


def policy_model(t_id, x):
    one_hot = tf.one_hot(t_id, 2)
    x = tf.concat([x, one_hot], axis=1)
    x = ks.layers.Dense(100, activation='elu')(x)
    x = ks.layers.Dense(100, activation='elu')(x)
    x = ks.layers.Dense(3, activation='softmax')(x)
    return x


def value_model(t_id, action, x):
    one_hot = tf.one_hot(t_id, 2)
    x = tf.concat([x, action, one_hot], axis=1)
    x = ks.layers.Dense(40, activation='elu')(x)
    x = ks.layers.Dense(20, activation='elu')(x)
    x = ks.layers.Dense(1, activation='linear')(x)
    return x


def policy_model2(t_id, x):
    x = ks.layers.Dense(100, activation='elu')(x)
    xs = []
    for i in range(4):
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
    for i in range(4):
        nx = ks.layers.Dense(100, activation='elu')(x)
        xs.append(ks.layers.Dense(1, activation='linear')(nx))
    xs = tf.stack(xs, axis=1)
    batch_indices = tf.range(tf.shape(t_id)[0])
    selectors = tf.stack([batch_indices, t_id], axis=1)
    return tf.gather_nd(xs, selectors)


env = lambda: wenv.GymEnvWrapper(gym.make('MountainCarLong-v0'), lambda s, a, r: np.array([r / 1000, np.abs(s[1])*10, 1 if a == 0 else 0, 1 if a == 2 else 0]), 4)
#env = lambda: wenv.GymEnvWrapper(gym.make('MountainCar-v0'), lambda s, a, r: np.array([r / 1000, np.abs(s[1])*10 if -0.6<s[0]<-0.4 and np.abs(s[1]) > 0.03 else 0]), 2)
#env = lambda: wenv.GymEnvWrapper(gym.make('MountainCar-v0'), lambda s, a, r: np.array([np.abs(s[1])*10, np.abs(s[1])*10]), 2)
sac_u = SacU(policy_model2, value_model2, env, (2,), [0, 1, 2], 4, 10, buffer_size=1000, visual=True, averaged_gradients=10,
             learning_rate=0.0001, entropy_regularization_factor=0.005, scheduler_period=200, gamma=0.9)
sac_u.run()