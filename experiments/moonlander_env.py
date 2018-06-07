import tensorflow as tf

from SAC_U.sac_u import SacU

ks = tf.keras
import environments.wrapper_env as wenv
import gym
import numpy as np
import environments.mountaincar_long


def policy_model2(t_id, x):
    x = ks.layers.Dense(100, activation='elu')(x)
    xs = []
    for i in range(1):
        nx = ks.layers.Dense(100, activation='elu')(x)
        xs.append(ks.layers.Dense(4, activation='softmax')(nx))
    xs = tf.stack(xs, axis=1)
    batch_indices = tf.range(tf.shape(t_id)[0])
    selectors = tf.stack([batch_indices, t_id], axis=1)
    return tf.gather_nd(xs, selectors)


def value_model2(t_id, action, x):
    x = tf.concat([x, action], axis=1)
    x = ks.layers.Dense(100, activation='elu')(x)
    xs = []
    for i in range(1):
        nx = ks.layers.Dense(100, activation='elu')(x)
        xs.append(ks.layers.Dense(1, activation='linear')(nx))
    xs = tf.stack(xs, axis=1)
    batch_indices = tf.range(tf.shape(t_id)[0])
    selectors = tf.stack([batch_indices, t_id], axis=1)
    return tf.gather_nd(xs, selectors)


env = lambda: wenv.GymEnvWrapper(gym.make('LunarLander-v2'), lambda s, a, r: np.array([r / 10]), 1)
#env = lambda: wenv.GymEnvWrapper(gym.make('MountainCar-v0'), lambda s, a, r: np.array([r / 1000, np.abs(s[1])*10 if -0.6<s[0]<-0.4 and np.abs(s[1]) > 0.03 else 0]), 2)
#env = lambda: wenv.GymEnvWrapper(gym.make('MountainCar-v0'), lambda s, a, r: np.array([np.abs(s[1])*10, np.abs(s[1])*10]), 2)
sac_u = SacU(policy_model2, value_model2, env, (8,), [0, 1, 2, 3], 1, 3, buffer_size=1000, visual=True, averaged_gradients=3,
             learning_rate=0.0001, entropy_regularization_factor=0.5, scheduler_period=200, gamma=0.99)
sac_u.run()