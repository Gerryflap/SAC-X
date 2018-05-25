"""
    The SAC-U actor collects trajectories and gives them to the learner
"""
import tensorflow as tf
import numpy as np


class SacUActor(object):
    def __init__(self, environment, n_trajectories, max_steps, scheduler_period, state_shape, action_space, n_tasks,
                 policy_model, learner, parameter_server):
        """
        Initializes the learner
        :param n_trajectories: Number of trajectories to be collected before submitting to learner.
        :param max_steps: Max steps taken before end of episode.
        :param scheduler_period: The number of steps we're staying on the same task before choosing another
        :param state_shape: The shape of the state variable (does not include batch dimension)
        :param action_space: The actions space, should be a list of objects that represent each action
        :param n_tasks: Number of tasks, both auxiliary and external
        :param policy_model: The policy model: takes taskID, observation (in batch);
            returns a |action_space| sized probability distribution
        """
        self.n_trajectories = n_trajectories  # N_trajectories
        self.max_steps = max_steps  # T
        self.scheduler_period = scheduler_period  # Xi
        self.policy_model = policy_model
        self.state_shape = state_shape
        self.action_space = action_space
        self.n_tasks = n_tasks
        self.state = tf.placeholder(tf.float32, state_shape)
        self.task_id = tf.placeholder(tf.int32)
        self.env = environment
        self.learner = learner
        self.parameter_server = parameter_server
        # Since this is SAC_U, there is no Q-table for scheduling

    def run(self):
        # Run the actor code, it's best to do this in a separate thread/process
        with tf.Session() as sess:
            with tf.variable_scope("policy"):
                policy = self.policy_model(self.task_id, self.state)
            init = tf.global_variables_initializer()
            sess.run(init)
            while True:
                # Keep working indefinitely
                n = 0
                task_id = None
                s = self.env.reset()
                trajectory = []
                while n < self.n_trajectories:
                    self.update_parameters(sess)

                    # Collect a new trajectory from the environment
                    for t in range(self.max_steps):
                        if t % self.scheduler_period == 0:
                            # Sample a uniformly random task:
                            task_id = np.random.randint(0, self.n_tasks)
                        action_dist = sess.run(
                            policy,
                            feed_dict={self.state: np.expand_dims(s, axis=0), self.task_id: task_id}
                        )[0]
                        a_i = self.sample_index(action_dist)
                        a = self.action_space[a_i]

                        # Here we assume that the environment will provide the rewards:
                        s_new, rewards = self.env.step(a)
                        trajectory.append((s, a_i, rewards, action_dist))
                    n += 1

                self.send_trajectory(trajectory)

    @staticmethod
    def sample_index(distribution):
        cum_prob = 0
        choice = np.random.random()
        for i, p in enumerate(distribution):
            cum_prob += p
            if cum_prob > choice:
                return i

        raise ValueError("Expected cumulative probability of 1, got %s. Failed to sample from distribution %s." % (
            np.sum(distribution), distribution))

    def update_parameters(self, sess):
        """
        Requests fresh parameters from the parameter server applies these new parameters
        :param sess: The TensorFlow session
        """
        variable_map = self.parameter_server.get_parameters()

        # Construct a "query" of reassignments to run on the session
        query = []
        for var in tf.trainable_variables("policy"):
            if var.name in variable_map:
                query.append(tf.assign(var.name, variable_map[var.name]))
        sess.run(query)

    def send_trajectory(self, trajectory):
        self.learner.add_trajectory(trajectory)
