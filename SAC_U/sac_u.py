"""
    The central class that starts everything. It will manage the startup of all the other classes.
"""

from SAC_U.parameter_server import SacUParameterServer
from SAC_U.learner import SacULearner
from SAC_U.actor import SacUActor


class SacU(object):
    def __init__(self, policy_model, value_model, environment, state_shape, action_space, n_tasks, n_learners,
                 learning_rate=0.0001, averaged_gradients=None, entropy_regularization_factor=0.1, n_trajectories=4,
                 max_steps = 4000, scheduler_period=150):
        self.entropy_regularization_factor = entropy_regularization_factor
        self.n_tasks = n_tasks
        self.action_space = action_space
        self.state_shape = state_shape
        self.environment = environment
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
                                  state_shape, action_space, n_tasks)
            actor = SacUActor(environment, n_trajectories, max_steps, scheduler_period, state_shape, action_space,
                              n_tasks, policy_model, learner, self.parameter_server)
            self.actors.append(actor)
            self.learners.append(learner)
            self.parameter_server.add_learner(learner)

    def run(self):
        # TODO: Run all threads. Run the parameter server on this thread
        pass
