"""
    The parameter server serves and updates parameters according to the incoming gradients.
"""
import multiprocessing as mp
import tensorflow as tf
import numpy as np


class SacUParameterServer(object):
    def __init__(self, value_model, policy_model, state_shape, action_space, gradients_to_average, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.gradients_to_average = gradients_to_average
        self.value_model = value_model
        self.policy_model = policy_model
        self.learners = []
        self.learner_barriers = []
        self.id = 0
        self.gradient_queue = mp.Queue()

        self.state = tf.placeholder(tf.float32, state_shape)
        self.task_id = tf.placeholder(tf.int32)
        self.action = tf.placeholder(tf.float32, shape=(len(action_space, )))
        self.parameters = None

    def add_learner(self, learner):
        """
        Adds a Learner to the list and increments the counter
        :param learner: The Learner to add
        """
        learner.learner_index = self.id
        self.id += 1
        self.learners.append(learner)

        # Add a barrier to hold the learner until parameters are updated
        self.learner_barriers.append(mp.Barrier(2))

    def run(self):
        with tf.Session() as sess:
            with tf.variable_scope("policy"):
                policy = self.policy_model(self.task_id, self.state)
            with tf.variable_scope("value"):
                value = self.value_model(self.task_id, self.action, self.state)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            init = tf.global_variables_initializer()
            sess.run(init)
            self.update_parameter_variable(sess)
            while True:
                n = 0
                d_theta, d_phi = [], []
                while n < self.gradients_to_average:
                    delta_theta, delta_phi = self.get_gradients()
                    d_theta.append(delta_theta)
                    d_phi.append(delta_phi)
                optimizer.apply_gradients(self.get_grads(d_phi))
                optimizer.apply_gradients(self.get_grads(d_theta))

    def get_grads(self, gradients: list) -> list:
        """
        Calculates the mean gradient for each variable
        :param gradients: A list of dicts of the form var_name -> gradient
        :return: Averaged gradients in the form [(grad_1, var_1), (grad_2, var_2), ...]
        """

        output = []
        for variable in tf.trainable_variables():
            sum_gradients = None
            for gradient_map in gradients:
                if variable.name in gradient_map:
                    gradient = gradient_map[variable.name]
                    if sum_gradients is None:
                        sum_gradients = gradient
                    else:
                        sum_gradients += gradient
            sum_gradients /= len(gradients)
            output.append((sum_gradients, variable))
        return output

    def update_parameter_variable(self, sess):
        # TODO: Implement. Should update parameter variable and unlock Learners
        pass

    def get_gradients(self):
        # TODO: Implement. Should return gradients from the queue and block if none are present
        pass
