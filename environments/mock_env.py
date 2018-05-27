import numpy as np

n_tasks = 3
state_shape = (3,)
action_space = [0,1,2]


class MockEnv(object):
    def __init__(self):
        pass

    def reset(self):
        return np.zeros(state_shape)

    def step(self, action):
        return np.zeros(state_shape), np.random.normal(0, 1, (n_tasks,))

