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
                 learning_rate=0.0001, averaged_gradients=None, entropy_regularization_factor=0.1,
                 max_steps = 4000, scheduler_period=150, buffer_size=-1, visual=False, gamma=1.0,
                 trajectory_listeners=None):
        if trajectory_listeners is None:
            trajectory_listeners = []
        self.trajectory_listeners = trajectory_listeners
        self.gamma = gamma
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
                                  state_shape, action_space, n_tasks, buffer_size=buffer_size, gamma=gamma)
            actor = SacUActor(environment_creator(), max_steps, scheduler_period, state_shape, action_space,
                              n_tasks, policy_model, learner, self.parameter_server, trajectory_listeners=trajectory_listeners)
            self.actors.append(actor)
            self.learners.append(learner)
            self.parameter_server.add_learner(learner)
            self.parameter_server.add_actor(actor)
        if visual:
            actor = SacUActor(environment_creator(), max_steps, scheduler_period, state_shape, action_space,
                              n_tasks, policy_model, None, self.parameter_server, visual=True,
                              trajectory_listeners=trajectory_listeners
                              )
            self.actors.append(actor)
            self.parameter_server.add_actor(actor)

    def run(self):
        processes = []
        for process in self.learners + self.actors:
            p1 = mp.Process(target=process.run)
            processes.append(p1)
            p1.start()

        for listener in self.trajectory_listeners:
            processes.append(listener)
            listener.start()
        try:
            self.parameter_server.run()
        except (KeyboardInterrupt, Exception):
            # Main process killed, terminate all subprocess
            traceback.print_exc()
            print("Terminating processes: ")
            for process in processes:
                process.terminate()
                print("Waiting for process: ", process.name)
                process.join()
                print("Terminated")


if __name__ == "__main__":
    import tensorflow as tf
    ks = tf.keras
    import environments.wrapper_env as wenv
    import gym
    import numpy as np
    from utility.avg_score_entropy_trajectory_listener import AvgScoreEntropyTrajectoryListener

    def policy_model(t_id, x):
        one_hot = tf.one_hot(t_id, 2)
        x = tf.concat([x, one_hot], axis=1)
        x = ks.layers.Dense(100, activation='elu')(x)
        x = ks.layers.Dense(100, activation='elu')(x)
        x = ks.layers.Dense(2, activation='softmax')(x)
        return x


    def value_model(t_id, action, x):
        one_hot = tf.one_hot(t_id, 2)
        x = tf.concat([x, action, one_hot], axis=1)
        x = ks.layers.Dense(40, activation='elu')(x)
        x = ks.layers.Dense(20, activation='elu')(x)
        x = ks.layers.Dense(1, activation='linear')(x)
        return x

    def policy_model2(t_id, x):
        x = ks.layers.Dense(10, activation='elu')(x)
        xs = []
        for i in range(1):
            nx = ks.layers.Dense(10, activation='elu')(x)
            xs.append(ks.layers.Dense(2, activation='softmax')(nx))
        xs = tf.stack(xs, axis=1)
        batch_indices = tf.range(tf.shape(t_id)[0])
        selectors = tf.stack([batch_indices, t_id], axis=1)
        return tf.gather_nd(xs, selectors)


    def value_model2(t_id, action, x):
        x = tf.concat([x, action], axis=1)
        x = ks.layers.Dense(10, activation='elu')(x)
        xs = []
        for i in range(1):
            nx = ks.layers.Dense(10, activation='elu')(x)
            xs.append(ks.layers.Dense(2, activation='linear')(nx))
        xs = tf.stack(xs, axis=1)
        batch_indices = tf.range(tf.shape(t_id)[0])
        selectors = tf.stack([batch_indices, t_id], axis=1)
        return tf.reduce_sum(tf.gather_nd(xs, selectors)*action, axis=1)

    listeners = [AvgScoreEntropyTrajectoryListener(200, [0], ['red'])]
    env = lambda: wenv.GymEnvWrapper(gym.make('CartPole-v0'), lambda s, a, r: np.array([r/100]), 1)
    sac_u = SacU(
        policy_model2,      # Policy neural net
        value_model2,       # Value neural net
        env,                # Environment creator function
        (4,),               # State shape
        [0,1],              # Action space
        1,                  # Number of tasks
        5,                 # Number of learners/actors
        buffer_size=10000,
        visual=False,
        averaged_gradients=5,
        learning_rate=2e-4,
        entropy_regularization_factor=0.0005,
        scheduler_period=200,
        gamma=0.3,
        max_steps=100,
        trajectory_listeners=listeners
    )
    sac_u.run()
