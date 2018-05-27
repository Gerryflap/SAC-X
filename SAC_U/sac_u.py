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
                 max_steps = 4000, scheduler_period=150, buffer_size=-1):
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

    def run(self):
        processes = []
        for learner, actor in zip(self.learners, self.actors):
            p1 = mp.Process(target=learner.run)
            p2 = mp.Process(target=actor.run)
            processes.append(p1)
            processes.append(p2)
            p1.start()
            p2.start()
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
    import environments.mock_env as mock_env

    def policy_model(t_id, x):
        x = ks.layers.Dense(10, activation='elu')(x)
        return ks.layers.Dense(len(mock_env.action_space), activation=ks.activations.softmax)(x)


    def value_model(t_id, action, x):
        x = tf.concat([x, action], axis=1)
        x = ks.layers.Dense(10, activation='elu')(x)
        return ks.layers.Dense(1, activation='linear')(x)


    env = lambda: mock_env.MockEnv()
    sac_u = SacU(policy_model, value_model, env, mock_env.state_shape, mock_env.action_space, mock_env.n_tasks, 2, buffer_size=1000)
    sac_u.run()