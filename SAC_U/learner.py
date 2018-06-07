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
        # Scale the term to account for the fact that TensorFlow can only do Ln(x) and not Log2(x)
        self.entropy_regularization = entropy_regularization/np.log(2)
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
                self.update_local_parameters(sess, both=True)
                n += 1

    def action_to_one_hot(self, action_index):
        """
        Converts an action to a one-hot vector with length |action_space|
        :param action_index: The index of the action in the one-hot vector
        :return: The one-hot vector as numpy array
        """
        v = np.zeros((len(self.action_space,)))
        v[action_index] = 1
        return v

    def update_replay_buffer(self):
        """
        Updates the replay buffer with incoming replays from the actor
        """
        while self.trajectory_queue.qsize() > 0:
            # While the queue is not empty: receive trajectories...
            trajectory = self.trajectory_queue.get()

            # ... and add them to the memory
            self.replay_buffer.append(trajectory)

            # Remove the first trajectory if the buffer is full:
            if self.buffer_size != -1 and len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)

    def update_local_parameters(self, sess, both=False):
        """
        Updates the local (to this learner) parameters using the parameters received from the server.
        This method blocks until parameters have been received,
            but will gladly take more parameters if available to get the most recent ones.
        :param sess: The TF session
        :param both: If set to True: updates both fixed and live/current networks
        """

        # Get a map of (variable name -> value) from the server
        variable_map = self.parameter_queue.get()

        # Or more if we're behind:
        while not self.parameter_queue.empty():
            variable_map = self.parameter_queue.get()

        # Construct a "query" of reassignments to run on the session
        feed_dict = dict()
        query = []

        # Loop over all local variables in the "current" scope
        for var in self.assigns.keys():

            # If the variable map contains this value,
            #   append this assignment to the query
            if var.name.replace("current/", "") in variable_map:

                # Get the TF assignment operator and new value placeholder from the dict
                assign_op, placeholder = self.assigns[var]

                # Add the assignment to the query
                query.append(assign_op)

                # Add the new value to the feed dict for the respective placeholder
                feed_dict[placeholder] = variable_map[var.name.replace("current/", "")]

        if both:
            # Do the same for the fixed variables:
            for var in self.assigns.keys():
                if var.name.replace("fixed/", "") in variable_map:
                    assign_op, placeholder = self.assigns[var]
                    query.append(assign_op)
                    feed_dict[placeholder] = variable_map[var.name.replace("fixed/", "")]

        # Run the query on the session
        sess.run(query, feed_dict=feed_dict)

    def get_qsa_values(self, states, tasks, sess):
        """
        Returns a numpy array of shape (len(states), len(action_space)) of (fixed) Q values for every action in every state
        :param sess: The TF session
        :param tasks: The task to select at every state
        :param states: The states to compute this on
        :return: The (fixed) Q-values for every action in action space for every state in states
        """
        policy, value, policy_fixed, value_fixed = self.parameters
        states_long = []
        tasks_long = []
        actions_long = []
        action_space_onehot = list([self.action_to_one_hot(a) for a in self.action_space])

        # Create a long batch of every possible combination of state and action
        for state, task_id in zip(states, tasks):
            states_long += [state]*len(self.action_space)
            tasks_long += [task_id] * len(self.action_space)
            actions_long += action_space_onehot

        # Get the Q-values
        q_values = sess.run(
            value_fixed,
            feed_dict={
                self.state: states_long,
                self.action: actions_long,
                self.task_id: tasks_long
            })

        # Reshape the outcome
        q_values = np.reshape(q_values, (-1, len(self.action_space)))
        return q_values

    def get_q_delta(self, trajectory, task_id,  sess):
        """
        Calculate the Q-delta according to the Retrace algorithm
            (the implementation is based on the Retrace paper, not the SAC-X paper)
        :param trajectory: The trajectory to calculate the delta on
        :param task_id: The task_id to calculate the delta for
        :param sess: The TF session
        :return: (The Q delta, the target q-value)
            Here the target q-value is the fixed q-values + the q-delta
        """
        policy, value, policy_fixed, value_fixed = self.parameters

        # Initialize the q_delta variable and get all Qsa values for the trajectory
        q_delta = 0
        q_values = self.get_qsa_values(
            [e[0] for e in trajectory],
            [task_id for e in trajectory],
            sess
        )

        # Calculate all values of π(* | state, task_id) for the trajectory for out current task
        #   (using the fixed network)
        policies = sess.run(
            policy_fixed,
            feed_dict={
                self.state: [e[0] for e in trajectory],
                self.task_id: [task_id for e in trajectory]
            })

        # Pick all π(a_t | state, task_id) from π(* | state, task_id) for every action taken in the trajectory
        all_action_probabilities = np.array([a[i] for i, a in zip([e[1] for e in trajectory], policies)])

        # Pick all b(a_t | state, B) from the trajectory
        all_b_action_probabilities = np.array(([experience[3][experience[1]] for experience in trajectory]))

        # Calculate the value of c_k for the whole trajectory
        c = all_action_probabilities / all_b_action_probabilities

        # Make sure that c is capped on 1, so c_k = min(1, c_k)
        c[c > 1] = 1

        # Keep the product of c_k values in a variable
        c_product = 1

        # Iterate over the trajectory to calculate the expected returns
        for j, (s, a, r, _, _) in enumerate(trajectory):

            # Check if we're at the end of the trajectory
            if j != len(trajectory) - 1:
                # If we're not: calculate the next difference and use the Q for j+1 as well

                # The Expected value of the Q(s_j+1, *) under the policy
                expected_q_tp1 = np.sum(policies[j + 1] * q_values[j + 1])

                # The delta for this lookahead
                delta = r[task_id] + self.gamma * expected_q_tp1 - q_values[j, a]
            else:
                # If this is the last entry, we'll assume the Q(s_j+1, *) to be fixed on 0 as the state is terminal
                delta = r[task_id] - q_values[j, a]

            # Add this to the sum of q_deltas, where the term is multiplied by gamma and delta
            q_delta += c_product * self.gamma ** j * delta

            # Multiply the c_product with the next c-value,
            #   this makes any move done after a "dumb" move (according to our policy) less significant
            c_product *= c[j]

        # Calculate the new "correct" Q-value for this s,a
        absolute_q = q_values[0, trajectory[0][1]] + q_delta
        return q_delta, absolute_q

    def get_delta_phi(self, trajectory, sess) -> dict:
        """
        Calculate the gradients for phi
        :param trajectory: A trajectory to calulate the values for
        :param sess: The TF session
        :return: A gradient dictionary
        """
        q_returns = []

        # Get the new Q-values for every task
        for task_id in range(self.n_tasks):
            q_returns.append([self.get_q_delta(trajectory, task_id, sess)[1]])

        # Calculate the gradients:
        gradients = sess.run(self.q_loss_grads, feed_dict={
            self.state: [trajectory[0][0]] * self.n_tasks,
            self.action: [self.action_to_one_hot(trajectory[0][1])] * self.n_tasks,
            self.task_id: list(range(self.n_tasks)),
            self.q_rets_t: q_returns
        })


        # Put the gradients in a dict:
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

