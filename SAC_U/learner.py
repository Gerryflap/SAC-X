"""
    The Learner in the SAC-U algorithm collects trajectories in the replay memory and computes gradients
    from the replay memory. These gradients are then sent to a central parameter server.
"""
import time

import tensorflow as tf
import numpy as np
import multiprocessing as mp


class SacULearner(object):
    def __init__(self, parameter_server, policy_model, value_model, entropy_regularization, state_shape, action_space,
                 n_tasks, learner_index=-1, buffer_size=-1, training_iterations=-1, gamma=1):

        self.learner_index = learner_index
        self.parameter_server = parameter_server
        self.policy_model = policy_model
        self.value_model = value_model
        self.entropy_regularization = entropy_regularization
        self.buffer_size = buffer_size
        self.training_iterations = training_iterations
        self.trajectory_queue = mp.Queue(500)
        self.state_shape = state_shape
        self.action_space = action_space
        self.n_tasks = n_tasks
        self.state = tf.placeholder(tf.float32, (None,) + state_shape)
        self.task_id = tf.placeholder(tf.int32, (None,))
        self.action = tf.placeholder(tf.float32, shape=(None, len(action_space,)))
        self.replay_buffer = []
        self.parameter_queue = mp.Queue()
        self.gamma = gamma

    def add_trajectory(self, trajectory):
        self.trajectory_queue.put(trajectory)
        print(self.trajectory_queue.qsize())

    def run(self):
        with tf.Session() as sess:
            with tf.variable_scope("current"):
                with tf.variable_scope("policy"):
                    policy = self.policy_model(self.task_id, self.state)
                with tf.variable_scope("value"):
                    value = self.value_model(self.task_id, self.action, self.state)
            with tf.variable_scope("fixed"):
                with tf.variable_scope("policy"):
                    policy_fixed = self.policy_model(self.task_id, self.state)
                with tf.variable_scope("value"):
                    value_fixed = self.value_model(self.task_id, self.action, self.state)
            parameters = policy, value, policy_fixed, value_fixed
            init = tf.global_variables_initializer()
            sess.run(init)

            n = 0

            # Wait until the first trajectories appear if the buffer is empty
            while len(self.replay_buffer) <= 1:
                self.update_replay_buffer()
                time.sleep(0.1)

            while n < self.training_iterations or self.training_iterations == -1:
                print("Updating replay buffer: ")
                self.update_replay_buffer()

                # TODO: Reset this back to 1000 like the paper suggests. I found this annoying since the actors provided many new experiences that were never used
                for k in range(10):
                    trajectory = self.replay_buffer[np.random.randint(0, len(self.replay_buffer)-1)]

                    # To generate a random s_i, we shorten the trajactory randomly.
                    # This is not done explicitly in the paper
                    trajectory = trajectory[np.random.randint(0, len(trajectory)-1):]

                    delta_phi = self.get_delta_phi(trajectory, parameters, sess)
                    delta_theta = self.get_delta_theta(trajectory, parameters, sess)

                    # The idea is that this will block until the parameter update is finished:
                    self.parameter_server.put_gradients(self.learner_index, delta_phi, delta_theta)

                    # Update the current parameters
                    if k != 9:
                        self.update_local_parameters(sess, both=False)

                # Update all parameters
                print("Updating all parameters")
                self.update_local_parameters(sess, both=True)
                print("Done updating all variables")

                n += 1

    def action_to_one_hot(self, action_index):
        v = np.zeros((len(self.action_space,)))
        v[action_index] = 1
        return v

    def update_replay_buffer(self):
        """
        Updates the replay buffer with incoming replays from the actor
        """
        while self.trajectory_queue.qsize() > 0:
            print("Queue not empty: receiving trajectories")
            trajectory = self.trajectory_queue.get()
            self.replay_buffer.append(trajectory)

            # Remove the first trajectory if the buffer is full:
            print(self.buffer_size, len(self.replay_buffer))
            if self.buffer_size != -1 and len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)
            print("Updated replay buffer, size: ", len(self.replay_buffer))
        print(self.trajectory_queue.empty(), self.trajectory_queue.qsize())

    def update_local_parameters(self, sess, both=False):

        variable_map = self.parameter_queue.get()
        while not self.parameter_queue.empty():
            variable_map = self.parameter_queue.get()

        # Construct a "query" of reassignments to run on the session
        query = []
        for var in tf.trainable_variables("current"):
            if var.name.replace("current/", "") in variable_map:
                query.append(tf.assign(var, variable_map[var.name.replace("current/", "")]))
        if both:
            for var in tf.trainable_variables("fixed"):
                if var.name.replace("fixed/", "") in variable_map:
                    query.append(tf.assign(var, variable_map[var.name.replace("fixed/", "")]))
        sess.run(query)

    def get_delta_phi(self, trajectory, parameters, sess) -> dict:
        # TODO: Generate the delta Phi
        # Stub implementation:

        policy, value, policy_fixed, value_fixed = parameters
        q_rets = []
        initial_experience = trajectory[0]
        for task_id in range(self.n_tasks):

            action_probabilities = sess.run(policy_fixed, feed_dict={
                self.task_id: [task_id],
                self.state: [initial_experience[0]]
            })[0]
            q_values = sess.run(value_fixed, feed_dict={
                self.task_id: [task_id]*len(self.action_space),
                self.state: [initial_experience[0]]*len(self.action_space),
                self.action: list([self.action_to_one_hot(a) for a in self.action_space])
            })
            print(q_values)
            q_values = np.reshape(q_values, (-1,))
            # The expected value of Q(s_i, . , T) over the policy distribution for s_i
            avg_q_si = np.sum(action_probabilities*q_values)

            all_q_values = sess.run(value_fixed, feed_dict={
                self.task_id: list([task_id for experience in trajectory]),
                self.state: list([experience[0] for experience in trajectory]),
                self.action: list([self.action_to_one_hot(experience[1]) for experience in trajectory])
            })
            q_deltas = avg_q_si - all_q_values
            all_action_distributions = sess.run(policy_fixed, feed_dict={
                self.task_id: list([task_id for experience in trajectory]),
                self.state: list([experience[0] for experience in trajectory]),
            })
            all_action_probabilities = np.sum(all_action_distributions* np.array([self.action_to_one_hot(experience[1]) for experience in trajectory]), axis=1)
            all_b_action_probabilities = np.array(([experience[3][experience[1]] for experience in trajectory]))
            c = all_action_probabilities/all_b_action_probabilities
            #print(c.shape, all_action_probabilities.shape, all_b_action_probabilities.shape)
            c[c>1] = 1

            rewards = np.array([experience[2][task_id] for experience in trajectory])

            q_ret = 0
            c_prod = 1
            for j in range(len(trajectory)):
                c_prod *= c[j]
                #print(c_prod, c[:j])
                q_ret = (self.gamma**j) * c_prod * (rewards[j] + q_deltas[j])
            q_rets.append(q_ret)
        q_rets = np.array(q_rets)
        q_rets = np.expand_dims(q_rets, axis=1)

        q_loss = tf.reduce_sum(np.square(value - q_rets))
        #print(q_rets)

        #print("Calculating gradients:")
        gradients = sess.run(tf.gradients(q_loss, tf.trainable_variables("current/value")),
                             feed_dict={
                                 self.task_id: list(range(self.n_tasks)),
                                 self.state: [initial_experience[0]]*self.n_tasks,
                                 self.action: [self.action_to_one_hot(initial_experience[1])]*self.n_tasks
                             })

        vars = tf.trainable_variables("current/value")
        values = gradients

        ret = dict()
        for var, val in zip(vars, values):
            ret[var.name.replace("current/", "")] = val

        return ret

    def get_delta_theta(self, trajectory, parameters, sess) -> dict:
        # TODO: Test this implementation

        policy, value, policy_fixed, value_fixed = parameters

        states = np.stack([event[0] for event in trajectory[:]], axis=0)
        # print("States: ", states)
        task_ids = list(range(self.n_tasks))
        values = np.zeros((self.n_tasks, len(states), len(self.action_space)))

        for task_id in task_ids:
            for action in range(len(self.action_space)):
                tasks = np.array([task_id] * states.shape[0])
                actions = np.repeat(np.expand_dims(self.action_to_one_hot(action), axis=0), states.shape[0], axis=0)

                values[task_id, :, action] = np.reshape(sess.run(value, feed_dict=
                {
                    self.state: states,
                    self.task_id: tasks,
                    self.action: actions
                }), (-1,))
        gradient_dict = dict()
        for task_id in range(self.n_tasks):
            task_policy_score = tf.reduce_sum(policy * (values[task_id] + self.entropy_regularization * tf.log(policy)))
            query = []
            keys = []
            for variable in tf.trainable_variables("current/policy"):
                query.append(tf.gradients(-task_policy_score, variable))
                keys.append(variable.name.replace("current/", ""))
            gradients = sess.run(
                query,
                feed_dict={
                    self.state: states,
                    self.task_id: np.array([task_id]*states.shape[0])
                })

            for key, grad in zip(keys, gradients):
                if key in gradient_dict:
                    gradient_dict[key] += grad[0]
                else:
                    gradient_dict[key] = grad[0]

        return gradient_dict

    def update_parameters(self, parameters):
        self.parameter_queue.put(parameters)

