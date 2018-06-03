import numpy as np

n_tasks = 1
state_shape = (3,)
action_space = [0,1,2]


class MockEnv(object):
    def __init__(self):
        self.terminated = False

    def reset(self):
        return np.random.normal(0, 1, state_shape)

    def step(self, action):
        rewards = np.zeros((n_tasks, ))
        if action == 0:
            rewards[0] = 0.001
        return np.random.normal(0, 1, state_shape), rewards

