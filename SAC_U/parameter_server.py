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
        self.actors = []
        self.learners = []
        self.id = 0
        self.gradient_queue = mp.Queue(gradients_to_average*2)

        self.state = tf.placeholder(tf.float32, (None,) + state_shape)
        self.task_id = tf.placeholder(tf.int32, (None,))
        self.action = tf.placeholder(tf.float32, shape=(None, len(action_space,)))
        self.parameters = dict()

    def add_learner(self, learner):
        """
        Adds a Learner to the list and increments the counter
        :param learner: The Learner to add
        """
        learner.learner_index = self.id
        self.id += 1
        self.learners.append(learner)

    def add_actor(self, actor):
        self.actors.append(actor)

    def run(self):
        with tf.Session() as sess:
            with tf.variable_scope("policy"):
                policy = self.policy_model(self.task_id, self.state)
            with tf.variable_scope("value"):
                value = self.value_model(self.task_id, self.action, self.state)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            grad_dict = dict()
            for var in tf.trainable_variables():
                grad_dict[var] = tf.placeholder(tf.float32, var.shape)
            grad_list = list([(placeholder, var) for var, placeholder in grad_dict.items()])


            # Apparently this is necessary to initialize Adam:
            optimize_op = optimizer.apply_gradients(grad_list)
            init = tf.global_variables_initializer()
            sess.run(init)
            self.update_parameter_variable(sess)
            while True:
                n = 0
                d_theta, d_phi = [], []
                while n < self.gradients_to_average:
                    delta_phi, delta_theta = self.get_gradients()
                    d_theta.append(delta_theta)
                    d_phi.append(delta_phi)
                    n += 1

                feed_dict = dict()
                for grads, var in self.get_grads(d_phi) + self.get_grads(d_theta):
                    feed_dict[grad_dict[var]] = grads

                sess.run([
                    optimize_op
                ], feed_dict=feed_dict)


                self.update_parameter_variable(sess)

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
            if sum_gradients is None:
                continue
            sum_gradients /= len(gradients)
            output.append((sum_gradients, variable))
        #print("Parameter output: ", output)
        return output

    def update_parameter_variable(self, sess):
        vars = tf.trainable_variables()
        vals = sess.run(tf.trainable_variables())

        params = dict()
        for var, val in zip(vars, vals):
            params[var.name] = val
        self.parameters = params
        #print(params)

        for actor in self.actors:
            actor.update_parameters(self.parameters)

        for learner in self.learners:
            learner.update_parameters(self.parameters)


    def get_gradients(self):
        return self.gradient_queue.get()

    def put_gradients(self, index, delta_phi, delta_theta):
        # This method will be executed by learners!
        self.gradient_queue.put((delta_phi, delta_theta))


