import random

import numpy as np


class SimpleTestEnv(object):
    def __init__(self):
        self.steps = 0
        self.terminated = True

    def reset(self):
        self.steps = 0
        self.terminated = False
        return np.array([0])

    def step(self, action):
        if self.steps == 200:
            self.terminated = True

        if action == 0:
            r = 0.01
        else:
            r = 0

        self.steps += 1

        return np.array([0], dtype=np.float32), np.array([r], dtype=np.float32)
