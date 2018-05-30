import time


class GymEnvWrapper(object):
    def __init__(self, gym_env, r_function, n_tasks):
        """

        :param gym_env: The gym environment
        :param r_function: A function that takes state, action and main reward and outputs a reward vector
        :param n_tasks: The size of this reward vector
        """
        self.n_tasks = n_tasks
        self.r_function = r_function
        self.gym_env = gym_env
        self.terminated = True
        self.render = False

    def reset(self):
        self.terminated = False
        return self.gym_env.reset()

    def set_rendering(self, rendering):
        self.render = rendering

    def step(self, action):
        s, r, d, _ = self.gym_env.step(action)
        if self.render:
            self.gym_env.render()
            time.sleep(1/60)
        self.terminated = d
        return s, self.r_function(s, action, r)