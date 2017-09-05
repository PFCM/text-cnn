"""
Various datasets for training the models at different levels.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.data as data


def load_vocab(sw_vocab):
    """
    Read a vocabulary and assign integer labels (lexicographically).
    Inserts a newline character into the vocab if it is not present.
    """
    with open(sw_vocab) as vocab_file:
        words = (line.split()[0] for line in vocab_file)
        vocab = {word: i for i, word in enumerate(sorted(words))}

    if '\n' not in vocab:
        vocab['\n'] = len(vocab)

    return vocab


def _parse_line(line):
    """
    Parse a line from a fastText .vec file.
    """
    items = line.split()
    word = items[0]
    if word == '</s>':
        word = '\n'

    vec = np.array([float(item) for item in line[1:]])

    return word, vec


def load_word_embeddings(ft_vecfile, vocab, trainable=False):
    """
    Load pre-trained word embeddings from the format output by fastText
    into a tensorflow variable.
    """
    with open(ft_vecfile) as ft_file:
        num_embeddings, embedding_dim = (int(item)
                                         for item
                                         in ft_file.readline().split())
        print('found {} embeddings of size {}'.format(num_embeddings,
                                                      embedding_dim))
        words_vectors = (_parse_line(line) for line in ft_file)
        int_to_vec = {vocab[word]: vec for word, vec in words_vectors}

    # stack them up
    embedding_array = np.array([int_to_vec[i] for i in range(num_embeddings)])

    embedding_matrix = tf.Variable(
        embedding_array,
        name='embeddings',
        trainable=trainable)

    return embedding_matrix


def _load_text(path, vocab):
    """
    Load the data as a big numpy array of ints.
    Tokenising simply splits on whitespace, although newlines are added back
    in.
    """
    with open(path) as text_file:
        tokens = (token
                  for line in text_file
                  for token in line.split() + ['\n'])
        return np.array([vocab[token] for token in tokens], dtype=np.int32)


def _regenerate_indices(data_length, batch_size):
    """
    generate a batch of indices, in the first quarter of the data
    """
    return np.random.randint(data_length//4, size=(batch_size,))


class TextDataset(object):
    """Class holding a bit of state for the dataset"""

    def __init__(self, flat_data, sequence_length, batch_size):
        """Set it up"""
        self._flat_data = flat_data
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self._batch_indices = _regenerate_indices(self._flat_data.shape[0],
                                                  self._batch_size)

    def get_batch(self):
        """Get some chunks of data"""
        data = np.array([self._flat_data[index:index+self._sequence_length]
                         for index in self._batch_indices])
        self._batch_indices += self._sequence_length
        if np.any(self._batch_indices > self._flat_data.shape[0]):
            self._batch_indices = _regenerate_indices(self._flat_data.shape[0],
                                                      self._batch_size)
        return data


def load_dataset(filepath, vocab, sequence_length, batch_size):
    """
    Get a tensor which will roll through batches of data.

    Each batch item goes through the text in order from a random
    starting position until one of them reaches the end.
    """
    # load the data into a big numpy array
    flat_data = _load_text(filepath, vocab)
    batch_indices = _regenerate_indices(flat_data.shape[0],
                                        batch_size)

    dataset = TextDataset(flat_data, sequence_length, batch_size)

    data_tensor = tf.py_func(dataset.get_batch, [], [tf.int32])[0]
    data_tensor.set_shape([batch_size, sequence_length])

    return data_tensor
