"""
    The SAC-U actor collects trajectories and gives them to the learner
"""
import tensorflow as tf
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

class SacUActor(object):
    def __init__(self, environment, n_trajectories, max_steps, scheduler_period, state_shape, action_space, n_tasks,
                 policy_model, learner, parameter_server, visual=False):
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
        self.state = tf.placeholder(tf.float32, (None,) + state_shape)
        self.task_id = tf.placeholder(tf.int32, (None,))
        self.env = environment
        self.learner = learner
        self.parameter_server = parameter_server
        self.parameter_queue = mp.Queue()
        self.visual = visual
        if visual:
            self.env.set_rendering(True)
            self.entropies = []
            self.scores = []
        self.assigns = dict()
        # Since this is SAC_U, there is no Q-table for scheduling

    def run(self):
        # Run the actor code, it's best to do this in a separate thread/process
        with tf.Session() as sess:
            with tf.variable_scope("policy"):
                policy = self.policy_model(self.task_id, self.state)
            init = tf.global_variables_initializer()
            sess.run(init)
            for var in tf.trainable_variables():
                placeholder = tf.placeholder(tf.float32, var.shape)
                self.assigns[var] = (tf.assign(var, placeholder), placeholder)
            print(self.assigns)

            while True:
                # Keep working indefinitely
                n = 0
                task_id = None
                s = self.env.reset()
                trajectory = []
                score = None
                while n < self.n_trajectories:
                    self.update_local_parameters(sess)

                    # Collect a new trajectory from the environment
                    for t in range(self.max_steps):
                        if t % self.scheduler_period == 0:
                            # Sample a uniformly random task:
                            task_id = np.random.randint(0, self.n_tasks)
                        action_dist = sess.run(
                            policy,
                            feed_dict={self.state: np.expand_dims(s, axis=0), self.task_id: np.array([task_id])}
                        )[0]
                        a_i = self.sample_index(action_dist)
                        a = self.action_space[a_i]

                        # Here we assume that the environment will provide the rewards:
                        s_new, rewards = self.env.step(a)
                        if score is None:
                            score = rewards
                        else:
                            score += rewards
                        trajectory.append((s, a_i, rewards, action_dist, task_id))
                        s = s_new
                        if self.env.terminated:
                            break
                    if self.env.terminated:
                        break
                    n += 1

                if not self.visual:
                    #print("Sending trajectory of length", len(trajectory), "with score ", score)
                    self.send_trajectory(trajectory)
                else:
                    self._visualize_trajectory(trajectory)

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


        # Construct a "query" of reassignments to run on the session
        query = []
        feed_dict = dict()
        for var in self.assigns.keys():
            if var.name in variable_map:
                assign_op, placeholder = self.assigns[var]
                query.append(assign_op)
                feed_dict[placeholder] = variable_map[var.name]
        sess.run(query, feed_dict=feed_dict)

    def send_trajectory(self, trajectory):
        self.learner.add_trajectory(trajectory)

    def update_parameters(self, parameters):
        self.parameter_queue.put(parameters)

    def _visualize_trajectory(self, trajectory):
        entropy = np.average([np.sum(step[3] * -np.log(step[3])) for step in trajectory])
        score = np.sum([step[2][0] for step in trajectory])
        self.scores.append(score)
        self.entropies.append(entropy)
        plt.plot(self.scores, color='red')
        plt.plot(self.entropies, color='blue')
        plt.pause(0.05)

