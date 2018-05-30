"""
    The central class that starts everything. It will manage the startup of all the other classes.
"""
import traceback

from SAC_U.parameter_server import SacUParameterServer
from SAC_U.learner import SacULearner
from SAC_U.actor import SacUActor
import multiprocessing as mp


class SacU(object):
    def __init__(self, policy_model, value_model, environment_creator, state_shape, action_space, n_tasks, n_learners,
                 learning_rate=0.0001, averaged_gradients=None, entropy_regularization_factor=0.1, n_trajectories=4,
                 max_steps = 4000, scheduler_period=150, buffer_size=-1, visual=False):
        self.buffer_size = buffer_size
        self.entropy_regularization_factor = entropy_regularization_factor
        self.n_tasks = n_tasks
        self.action_space = action_space
        self.state_shape = state_shape
        self.environment_creator = environment_creator
        self.policy_model = policy_model
        self.value_model = value_model
        self.learning_rate = learning_rate
        if averaged_gradients is None:
            averaged_gradients = n_learners

        self.parameter_server = SacUParameterServer(value_model, policy_model, state_shape, action_space,
                                                    averaged_gradients, learning_rate)
        self.learners = []
        self.actors = []
        for i in range(n_learners):
            learner = SacULearner(self.parameter_server, policy_model, value_model, entropy_regularization_factor,
                                  state_shape, action_space, n_tasks, buffer_size=buffer_size)
            actor = SacUActor(environment_creator(), n_trajectories, max_steps, scheduler_period, state_shape, action_space,
                              n_tasks, policy_model, learner, self.parameter_server)
            self.actors.append(actor)
            self.learners.append(learner)
            self.parameter_server.add_learner(learner)
            self.parameter_server.add_actor(actor)
        if visual:
            actor = SacUActor(environment_creator(), n_trajectories, max_steps, scheduler_period, state_shape, action_space,
                              n_tasks, policy_model, None, self.parameter_server, visual=True)
            self.actors.append(actor)
            self.parameter_server.add_actor(actor)

    def run(self):
        processes = []
        for process in self.learners + self.actors:
            p1 = mp.Process(target=process.run)
            processes.append(p1)
            p1.start()
        try:
            self.parameter_server.run()
        except Exception:
            # Main process killed, terminate all subprocess
            traceback.print_exc()
            for process in processes:
                process.terminate()

if __name__ == "__main__":
    import tensorflow as tf
    ks = tf.keras
    import environments.wrapper_env as wenv
    import gym
    import numpy as np

    def policy_model(t_id, x):
        x = ks.layers.Dense(100, activation='elu')(x)
        x = ks.layers.Dense(100, activation='elu')(x)
        return ks.layers.Dense(2, activation=ks.activations.softmax)(x)


    def value_model(t_id, action, x):
        x = tf.concat([x, action], axis=1)
        x = ks.layers.Dense(100, activation='elu')(x)
        x = ks.layers.Dense(100, activation='elu')(x)
        return ks.layers.Dense(1, activation='linear')(x)


    env = lambda: wenv.GymEnvWrapper(gym.make('CartPole-v0'), lambda s, a, r: np.array([r]), 1)
    sac_u = SacU(policy_model, value_model, env, (4,), [0,1], 1, 4, buffer_size=10000, visual=True, averaged_gradients=1, learning_rate=0.0001, entropy_regularization_factor=0.001)
    sac_u.run()