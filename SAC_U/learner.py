"""
    The Learner in the SAC-U algorithm collects trajectories in the replay memory and computes gradients
    from the replay memory. These gradients are then sent to a central parameter server.
"""
import tensorflow as tf
import numpy as np
import multiprocessing as mp


class SacULearner(object):
    def __init__(self, parameter_server, policy_model, value_model, entropy_regularization, state_shape, action_space,
                 n_tasks, buffer_size=-1, training_iterations=-1):

        self.parameter_server = parameter_server
        self.policy_model = policy_model
        self.value_model = value_model
        self.entropy_regularization = entropy_regularization
        self.buffer_size = buffer_size
        self.training_iterations = training_iterations
        self.trajectory_queue = mp.Queue()
        self.state_shape = state_shape
        self.action_space = action_space
        self.n_tasks = n_tasks
        self.state = tf.placeholder(tf.float32, state_shape)
        self.task_id = tf.placeholder(tf.int32)
        self.action = tf.placeholder(tf.float32, shape=(len(action_space,)))
        self.replay_buffer = []

    def add_trajectory(self, trajectory):
        self.trajectory_queue.put(trajectory)

    def run(self):
        with tf.Session() as sess:
            with tf.variable_scope("policy"):
                policy = self.policy_model(self.task_id, self.state)
            with tf.variable_scope("value"):
                value = self.value_model(self.task_id, self.action, self.state)
            init = tf.global_variables_initializer()
            sess.run(init)

            n = 0
            while n < self.n_tasks or self.n_tasks == -1:
                self.update_replay_buffer()

    def action_to_one_hot(self, action_index):
        v = np.zeros((len(self.action_space,)))
        v[action_index] = 1
        return v

    def update_replay_buffer(self):
        while not self.trajectory_queue.empty():
            self.replay_buffer.append(self.trajectory_queue.get(0))

            # Remove the first trajectory if the buffer is full:
            if self.buffer_size != -1 and len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)
