"""
Autoregressive RNN baseline.
Potentially with causal convolutions for extra goodness or something.
"""
import tensorflow as tf
from tensorflow.contrib.framework import nest


def rnn_model(inputs, shape, embedding_matrix):
    """make an unrolled RNN over the inputs. Not optimised for GPU"""
    with tf.variable_scope('rnn'):
        inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
        input_shape = inputs.get_shape().as_list()
        vocab_size = embedding_matrix.get_shape()[0].value

        cells = [tf.nn.rnn_cell.GRUCell(n) for n in shape]
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        # won't work with LSTMs
        initial_state = tuple(
            tf.get_variable(
                'state_{}'.format(i),
                shape=[input_shape[0], c.state_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())
            for i, c in enumerate(cells))
        outputs, final_state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=initial_state)
        # we're always going to roll this over every time the output is
        # evaluated
        state_updates = nest.map_structure(tf.assign, initial_state,
                                           final_state)
        state_updates = nest.flatten(state_updates)
        with tf.control_dependencies(state_updates):
            outputs = tf.reshape(outputs, [-1, shape[-1]])
            outputs = tf.layers.dense(outputs, vocab_size, activation=None)
            outputs = tf.reshape(
                outputs, [input_shape[0] or -1, input_shape[1], vocab_size])
        return outputs


class RNNModel(tf.keras.Model):
    """baseline rnn"""

    def __init__(self, vocab_size, num_units, embedding_matrix):
        """get ready to go"""
        super(RNNModel, self).__init__()
        self.num_units = num_units

        self.embedding_matrix = embedding_matrix

        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(
                self.num_units,
                return_sequences=True,
                recurrent_initializer='orthogonal',
                stateful=True)
        else:
            self.gru = tf.keras.layers.GRU(
                self.num_units,
                return_sequences=True,
                recurrent_activation='sigmoid',
                recurrent_initializer='orthogonal',
                stateful=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        """embed the inputs then push through the net"""
        inputs = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return self.fc(self.gru(inputs))
