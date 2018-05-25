import tensorflow as tf
import numpy as np
import multiprocessing as mp
ks = tf.keras


def model(inp):
    x = ks.layers.Dense(5, activation='tanh')(inp)
    return ks.layers.Dense(1, activation='tanh')(inp)


queue = mp.Queue(1)


def trainer():
    with tf.Session() as sess:
        inp = tf.constant(np.array([[1]]), dtype=tf.float32)
        out = model(inp)
        optimizer = tf.train.AdamOptimizer(0.001).minimize(out)

        init = tf.global_variables_initializer()
        sess.run(init)

        print("Trainer output before training: ", sess.run(out))

        for i in range(1000):
            sess.run(optimizer)
        print("Trainer output after training: ", sess.run(out))

        variables = tf.trainable_variables()
        values = sess.run(variables)

        variables = [var.name for var in variables]

        value_map = dict()
        for var, val in zip(variables, values):
            value_map[var] = val

        queue.put(value_map)



def waiter():
    with tf.Session() as sess:
        inp = tf.constant(np.array([[1]]), dtype=tf.float32)
        out = model(inp)

        init = tf.global_variables_initializer()
        sess.run(init)

        print("Waiter output before training: ", sess.run(out))
        value_map = queue.get()
        query = []
        for variable in tf.trainable_variables():
            if variable.name in value_map:
                query.append(tf.assign(variable, value_map[variable.name]))
        sess.run(query)

        print("Waiter output after training: ", sess.run(out))


if __name__ == "__main__":
    mp.Process(target=trainer).start()
    waiter()