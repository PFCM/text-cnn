"""
Train a model, either to generate or upsample.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import read
import models


def _downsample_inputs(data, downsample):
    """reduce in time by averaging"""
    data = tf.expand_dims(data, 1)
    data = tf.nn.avg_pool(data,
                          [1, 1, downsample, 1],
                          [1, 1, downsample, 1]
                          'SAME')
    return data


def get_data(data_path,
             vocab_path,
             embedding_path,
             downsampling,
             sequence_length,
             batch_size):
    """get data, embed it and potentially downsample it"""
    with tf.variable_scope('data'):
        vocab = read.load_vocab(vocab_path)
        data_tensor = read.load_dataset(data_path,
                                        vocab,
                                        sequence_length,
                                        batch_size)
    with tf.variable_scope('embeddings'):
        embedding_matrix = read.load_word_embeddings(embedding_path, vocab)
        embedded_data = tf.nn.embedding_lookup(embedding_matrix, data_tensor)
        if downsampling > 1:
            embedded_data = _downsample_inputs(embedded_data, downsample)
    return embedded_data, vocab



def main(_):
    """actually do a things"""
    pass


if __name__ == '__main__':
    tf.app.run()
