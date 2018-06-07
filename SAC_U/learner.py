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
        """
        Initializes a SAC-U Learner
        :param parameter_server: The central parameter server
        :param policy_model: The policy model: takes taskIDs, observations (both in batch);
            returns a |action_space| sized probability distribution (in batch)
        :param value_model: The value model: takes taskIDs, actions (one hot) and observations in batch;
            returns a (-1, 1) shaped Tensor (where the -1 is flexible depending on the input length) of q-values.
        :param entropy_regularization: The entropy regularization factor as described in equation 9 in the paper;
            Do note that the paper uses "+ α * log π(a | s_t, T)" whereas this implementation uses
            - α * log π(a | s_t, T) because it makes more sense in my opinion.
            A higher α in this implementation keeps the entropy higher.
        :param state_shape: The shape of the state variable (does not include batch dimension)
        :param action_space: The actions space, should be a list of objects that represent each action
        :param n_tasks: Number of tasks, both auxiliary and external
        :param learner_index: The index of the learner, currently not really used.
        :param buffer_size: The size of the replay buffer (in trajectories)
        :param training_iterations: Number of training iterations the actor has to perform. -1 denotes infinitely long
        :param gamma: The discount factor γ, should be between 0 and 1.
            γ = 0 makes the agent only care about immediate rewards.
            γ = 1 makes every action "responsible" for all future rewards, even if they happened very far in the future
        """

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
        self.task_policy_score = None
        self.values_t = None
        self.q_loss_t = None
        self.q_loss_grads = None
        self.q_rets_t = None
        self.policy_grad = None
        self.assigns = dict()
        self.parameters = None

    def add_trajectory(self, trajectory):
        # This method is used by the actors to add trajectories
        self.trajectory_queue.put(trajectory)

    def run(self):

        # Initialize the TF session
        with tf.Session() as sess:

            # Build "live"/"current" policy and value networks
            with tf.variable_scope("current"):
                with tf.variable_scope("policy"):
                    policy = self.policy_model(self.task_id, self.state)
                with tf.variable_scope("value"):
                    value = self.value_model(self.task_id, self.action, self.state)

            # Build "fixed" target networks for policy and value
            with tf.variable_scope("fixed"):
                with tf.variable_scope("policy"):
                    policy_fixed = self.policy_model(self.task_id, self.state)
                with tf.variable_scope("value"):
                    value_fixed = self.value_model(self.task_id, self.action, self.state)
            parameters = policy, value, policy_fixed, value_fixed
            self.parameters = parameters

            # A tensor that holds the q-values for every action in the action_space, used for policy loss
            self.values_t = tf.placeholder(tf.float32, (None, len(self.action_space)))

            # The policy score. Since Adam uses gradient descend, the score is inverted to make it a loss function.
            self.task_policy_score = -tf.reduce_sum(policy * (self.values_t - self.entropy_regularization * tf.log(policy)))

            # The q-returns placeholder,
            #   holds the expected future rewards that the value network has to be optimized towards
            self.q_rets_t = tf.placeholder(tf.float32, (None, 1))

            # The MSE loss function for optimizing the value network
            self.q_loss_t = tf.reduce_sum(tf.square(value - self.q_rets_t))

            # The TF Ops for retrieving the gradients
            self.q_loss_grads = tf.gradients(self.q_loss_t, tf.trainable_variables("current/value"))
            self.policy_grad = tf.gradients(self.task_policy_score, tf.trainable_variables("current/policy"))

            # Create assign placeholders for assigning the variables TODO: using tf.Variable.load is better
            for var in tf.trainable_variables():
                placeholder = tf.placeholder(tf.float32, var.shape)
                self.assigns[var] = (tf.assign(var, placeholder), placeholder)

            init = tf.global_variables_initializer()
            sess.run(init)

            n = 0

            # Wait until the first 10 trajectories appear if the buffer is empty
            while len(self.replay_buffer) <= 10:
                self.update_replay_buffer()
                time.sleep(0.1)

            while n < self.training_iterations or self.training_iterations == -1:
                # Receive trajectories from the Learner
                self.update_replay_buffer()

                # Calculate 1000 gradients without updating the target networks
                for k in range(1000):
                    trajectory = self.replay_buffer[np.random.randint(0, len(self.replay_buffer)-1)]

                    # To generate a random s_i, we shorten the trajectory randomly.
                    # This is not done explicitly in the paper
                    trajectory = trajectory[np.random.randint(0, len(trajectory)-1):]

                    # Calculate the gradients that have to be supplied to the parameter server
                    delta_phi = self.get_delta_phi(trajectory, sess)
                    delta_theta = self.get_delta_theta(trajectory, sess)

                    # Push the gradients to the parameter server
                    self.parameter_server.put_gradients(self.learner_index, delta_phi, delta_theta)

                    # Update the current parameters
                    #   (unless it is the last iteration, then quit the loop and update both fixed and current networks)
                    if k != 999:
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
        feed_dict = dict()
        query = []
        for var in self.assigns.keys():
            if var.name.replace("current/", "") in variable_map:
                assign_op, placeholder= self.assigns[var]
                query.append(assign_op)
                feed_dict[placeholder] = variable_map[var.name.replace("current/", "")]
                #query.append(tf.assign(var, variable_map[var.name.replace("current/", "")]))
        if both:
            for var in self.assigns.keys():
                if var.name.replace("fixed/", "") in variable_map:
                    assign_op, placeholder = self.assigns[var]
                    query.append(assign_op)
                    feed_dict[placeholder] = variable_map[var.name.replace("fixed/", "")]
        sess.run(query, feed_dict=feed_dict)

    def get_qsa_values(self, states, tasks, sess):
        """
        Returns a numpy array of shape (len(states), len(action_space)) of Q values for every action in every state
        :param sess: The TF session
        :param tasks: The task to select at every state
        :param states: The states to compute this on
        :return: The Q-values for every action in action space for every state in states
        """
        policy, value, policy_fixed, value_fixed = self.parameters
        states_long = []
        tasks_long = []
        actions_long = []
        action_space_onehot = list([self.action_to_one_hot(a) for a in self.action_space])
        for state, task_id in zip(states, tasks):
            states_long += [state]*len(self.action_space)
            tasks_long += [task_id] * len(self.action_space)
            actions_long += action_space_onehot

        q_values = sess.run(
            value_fixed,
            feed_dict={
                self.state: states_long,
                self.action: actions_long,
                self.task_id: tasks_long
            })
        q_values = np.reshape(q_values, (-1, len(self.action_space)))
        return q_values



    def get_q_delta(self, trajectory, task_id,  sess):
        policy, value, policy_fixed, value_fixed = self.parameters


        q_delta = 0
        q_values = self.get_qsa_values(
            [e[0] for e in trajectory],
            [task_id for e in trajectory],
            sess
        )
        policies = sess.run(
            policy_fixed,
            feed_dict={
                self.state: [e[0] for e in trajectory],
                self.task_id: [task_id for e in trajectory]
            })

        all_action_probabilities = np.array([a[i] for i, a in zip([e[1] for e in trajectory], policies)])
        all_b_action_probabilities = np.array(([experience[3][experience[1]] for experience in trajectory]))
        c = all_action_probabilities / all_b_action_probabilities
        # print(c.shape, all_action_probabilities.shape, all_b_action_probabilities.shape)
        c[c > 1] = 1

        c_product = 1
        for j, (s, a, r, _, _) in enumerate(trajectory):
            if j != len(trajectory) - 1:
                expected_q_tp1 = np.sum(policies[j + 1] * q_values[j + 1])
                delta = r[task_id] + self.gamma * expected_q_tp1 - q_values[j, a]
            else:
                delta = r[task_id] - q_values[j, a]
            q_delta += c_product * self.gamma ** j * delta
            c_product *= c[j]

        absolute_q = q_values[0, trajectory[0][1]] + q_delta
        return q_delta, absolute_q

    def get_delta_phi(self, trajectory, sess) -> dict:
        q_returns = []
        for task_id in range(self.n_tasks):
            q_returns.append([self.get_q_delta(trajectory, task_id, sess)[1]])
        gradients = sess.run(self.q_loss_grads, feed_dict={
            self.state: [trajectory[0][0]] * self.n_tasks,
            self.action: [self.action_to_one_hot(trajectory[0][1])] * self.n_tasks,
            self.task_id: list(range(self.n_tasks)),
            self.q_rets_t: q_returns
        })

        vars = tf.trainable_variables("current/value")
        values = gradients

        ret = dict()
        for var, val in zip(vars, values):
            ret[var.name.replace("current/", "")] = val
        return ret


    def get_delta_phi_old(self, trajectory, sess) -> dict:
        # THIS METHOD IS DEPRECATED

        policy, value, policy_fixed, value_fixed = self.parameters
        q_rets = []
        initial_experience = trajectory[0]

        # TODO: This whole part should be updated:
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
            q_values = np.reshape(q_values, (-1,))
            # The expected value of Q(s_i, . , T) over the policy distribution for s_i
            avg_q_si = np.sum(action_probabilities*q_values)

            all_q_values = sess.run(value_fixed, feed_dict={
                self.task_id: list([task_id for experience in trajectory]),
                self.state: list([experience[0] for experience in trajectory]),
                self.action: list([self.action_to_one_hot(experience[1]) for experience in trajectory])
            })
            q_deltas = avg_q_si - all_q_values

            all_action_q_values = np.zeros((len(trajectory), len(self.action_space)))
            for action in self.action_space:
                all_action_q_values[:, action] = sess.run(value_fixed, feed_dict={
                    self.task_id: list([task_id for experience in trajectory]),
                    self.state: list([experience[0] for experience in trajectory]),
                    self.action: list([self.action_to_one_hot(action) for experience in trajectory])
                })[0]

            all_action_distributions = sess.run(policy_fixed, feed_dict={
                self.task_id: list([task_id for experience in trajectory]),
                self.state: list([experience[0] for experience in trajectory]),
            })
            print("AAD.shape:", all_action_distributions.shape)
            all_action_probabilities = np.sum(all_action_distributions* np.array([self.action_to_one_hot(experience[1]) for experience in trajectory]), axis=1)
            all_b_action_probabilities = np.array(([experience[3][experience[1]] for experience in trajectory]))
            c = all_action_probabilities/all_b_action_probabilities
            #print(c.shape, all_action_probabilities.shape, all_b_action_probabilities.shape)
            c[c>1] = 1

            rewards = np.array([experience[2][task_id] for experience in trajectory])

            q_ret = 0
            c_prod = 1
            c_reward = 0
            print("Deltas: ", q_deltas[:10, 0])
            #print("Reward: ", np.array([experience[2][task_id] for experience in trajectory])[:10])
            #print("Tasks: ", np.array([experience[4] for experience in trajectory])[:10])

            for j in range(len(trajectory)-1):
                c_reward += (self.gamma**j) * rewards[j]
                # According to the RETRACE paper j=0 should yield c_prod == 1
                if j != 0:
                    c_prod *= c[j]
                #print(c_prod, c[:j])
                #print(j, len(trajectory), all_action_distributions.shape, all_action_q_values.shape)
                expected_q_tp1 = np.sum(all_action_distributions[j+1] * all_action_q_values[j+1])
                q_ret += c_prod * (c_reward + self.gamma**(j+1) * expected_q_tp1 - all_q_values[j])
            # TODO: Remove MC:
            #q_ret = [np.sum(rewards)]
            q_rets.append(q_ret + all_q_values[0])

        q_rets = np.array(q_rets)
        #print(q_rets)
        #print("Initial action: ", initial_experience[1])
        #print(self.q_loss_grads, self.q_loss_t, value)

        gradients, q_loss, value_v = sess.run((self.q_loss_grads, self.q_loss_t, value),
                             feed_dict={
                                 self.task_id: list(range(self.n_tasks)),
                                 self.state: [initial_experience[0]]*self.n_tasks,
                                 self.action: [self.action_to_one_hot(initial_experience[1])]*self.n_tasks,
                                 self.q_rets_t: q_rets
                             })
        #print("Values, returns: ", value_v, q_rets)
        #print("Q-loss: ", q_loss)
        vars = tf.trainable_variables("current/value")
        values = gradients

        ret = dict()
        for var, val in zip(vars, values):
            ret[var.name.replace("current/", "")] = val

        return ret

    def get_delta_theta(self, trajectory, sess) -> dict:
        # TODO: Test this implementation

        policy, value, policy_fixed, value_fixed = self.parameters

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
        #print("Values: ", values)
        for task_id in range(self.n_tasks):
            query = []
            keys = []
            # TODO: Change the gradient calculation and move it out of here
            for variable in tf.trainable_variables("current/policy"):
                keys.append(variable.name.replace("current/", ""))
            gradients = sess.run(
                self.policy_grad,
                feed_dict={
                    self.state: states,
                    self.task_id: np.array([task_id]*states.shape[0]),
                    self.values_t: values[task_id]
                })

            for key, grad in zip(keys, gradients):
                if key in gradient_dict:
                    gradient_dict[key] += grad
                else:
                    gradient_dict[key] = grad
        return gradient_dict

    def update_parameters(self, parameters):
        self.parameter_queue.put(parameters)

