"""
    The SAC-U actor collects trajectories and gives them to the learner and trajectory listeners
"""
import queue
import time

import tensorflow as tf
import numpy as np
import multiprocessing as mp


class SacUActor(object):
    def __init__(self, environment, max_steps, scheduler_period, state_shape, action_space, n_tasks,
                 policy_model, learner, parameter_server, visual=False, trajectory_listeners=None):
        """
        Initializes the learner
        :param environment: The environment to operate on
        :param max_steps: Max steps taken before end of episode
        :param scheduler_period: The number of steps we're staying on the same task before choosing another
        :param state_shape: The shape of the state variable (does not include batch dimension)
        :param action_space: The actions space, should be a list of objects that represent each action
        :param n_tasks: Number of tasks, both auxiliary and external
        :param policy_model: The policy model: takes taskID, observation (both in batch);
            returns a |action_space| sized probability distribution (in batch)
        :param learner: The Learner attached to this actor, visual actors don't use a learner
        :param parameter_server: The central parameter server
        :param visual: A boolean flag that can enable visual mode.
            In visual mode the actor will render the environment
        :param trajectory_listeners: A list of trajectory listeners.
            Trajectory Listeners will receive the trajectories for monitoring purposes
        """
        if trajectory_listeners is None:
            trajectory_listeners = []
        self.trajectory_listeners = trajectory_listeners
        self.max_steps = max_steps  # T
        self.scheduler_period = scheduler_period  # Xi
        self.policy_model = policy_model
        self.state_shape = state_shape
        self.action_space = action_space
        self.n_tasks = n_tasks
        self.state = tf.placeholder(tf.float32, (None,) + state_shape)
        self.task_id = tf.placeholder(tf.int32, (None,))
        self.env = environment
        self.learner = learner
        self.parameter_server = parameter_server
        self.parameter_queue = mp.Queue()
        self.visual = visual
        self.env.render = self.visual
        self.assigns = dict()
        self.policy = None
        # Since this is SAC_U, there is no Q-table for scheduling

    def run(self):
        # Run the actor code, it's best to do this in a separate thread/process

        # Start the TensorFlow session
        with tf.Session() as sess:

            # Define the policy network
            with tf.variable_scope("policy"):
                policy = self.policy_model(self.task_id, self.state)
            self.policy = policy

            # Initialize all variables
            init = tf.global_variables_initializer()
            sess.run(init)

            # Create the assignment operations
            for var in tf.trainable_variables():
                placeholder = tf.placeholder(tf.float32, var.shape)
                self.assigns[var] = (tf.assign(var, placeholder), placeholder)

            task_id = None
            while True:
                # Keep working indefinitely

                s = self.env.reset()
                trajectory = []

                # Fetch parameters
                self.update_local_parameters(sess)
                # Collect a new trajectory from the environment
                for t in range(self.max_steps):

                    # If t mod xi == 0, switch the task to a uniformly random sampled task
                    if t % self.scheduler_period == 0:
                        # Sample a uniformly random task:
                        task_id = np.random.randint(0, self.n_tasks)

                    # Get π(* | state = s, task_id)
                    action_dist = self.get_action_distribution(s, task_id, sess)

                    # Sample an action index from π(* | s, task_id)
                    a_i = self.sample_index(action_dist)

                    # Retrieve the action that corresponds with this action id
                    a = self.action_space[a_i]

                    # Collect the new state and reward vector from the environment
                    s_new, rewards = self.env.step(a)


                    # Append the experience to the trajectory
                    trajectory.append((s, a_i, rewards, action_dist, task_id))

                    # Update the current state and check for environment termination
                    s = s_new
                    if self.env.terminated:
                        break

                if not self.visual:
                    self.send_trajectory(trajectory)

    def get_action_distribution(self, state, task_id, sess):
        """
        Calculates π(* | state, task_id)
        :param state: the state, a numpy array of shape self.state_shape
        :param task_id: An integer denoting the current task
        :param sess: The TensorFlow session
        :return: π(* | state, task_id) (so a probability vector with a probability for every action in self.action_space)
        """
        action_dist = sess.run(
            self.policy,
            feed_dict={self.state: np.expand_dims(state, axis=0), self.task_id: np.array([task_id])}
        )[0]
        return action_dist


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

    def update_local_parameters(self, sess):
        """
        Gets fresh parameters from the parameter server applies these new parameters
        :param sess: The TensorFlow session
        """

        if self.parameter_queue.empty():
            return
        variable_map = self.parameter_queue.get()
        while not self.parameter_queue.empty():
            variable_map = self.parameter_queue.get()

        query = []
        feed_dict = dict()
        for var in self.assigns.keys():
            if var.name in variable_map:
                assign_op, placeholder = self.assigns[var]
                query.append(assign_op)
                feed_dict[placeholder] = variable_map[var.name]
        sess.run(query, feed_dict=feed_dict)

    def send_trajectory(self, trajectory):
        for listener in self.trajectory_listeners:
            listener.put_trajectory(trajectory)

        done = False
        while not done:
            try:
                self.learner.add_trajectory(trajectory)
                done = True
            except queue.Full:
                print("Sending trajectory failed")
                pass

    def update_parameters(self, parameters):
        done = False
        while not done:
            try:
                self.parameter_queue.put(parameters, timeout=10)
                done = True
            except queue.Full:
                print("Waiting parameter update")
                pass


