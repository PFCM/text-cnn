"""
Various datasets for training the models at different levels.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from six.moves import reduce


def load_vocab(sw_vocab):
    """
    Read a vocabulary and assign integer labels (lexicographically).
    Inserts a newline character into the vocab if it is not present.
    """
    with open(sw_vocab) as vocab_file:
        words = (line.split(' ')[0] for line in vocab_file)
        words = list(words)
        vocab = {word: i for i, word in enumerate(sorted(words))}
    if '\n' not in vocab:
        vocab['\n'] = len(vocab)

    print('\n'.join(
        '{}: {}'.format(*pair)
        for pair in list(sorted(vocab.items(), key=lambda p: p[1]))[:10]))
    return vocab


def _parse_line(line):
    """
    Parse a line from a fastText .vec file.
    """
    items = line.rstrip().split(' ')
    word = items[0]
    if word == '</s>':
        word = '\n'
    elif word == '@@':
        word = '\xa0@@'

    vec = np.array([float(item) for item in items[1:]], dtype=np.float32)

    return word, vec


def _load_fasttext_embeddings(ft_vecfile, vocab, trainable=False):
    """
    Load pre-trained word embeddings from the format output by fastText
    into a tensorflow variable.
    """
    with open(ft_vecfile) as ft_file:
        num_embeddings, embedding_dim = (
            int(item) for item in ft_file.readline().split())
        print('found {} embeddings of size {}'.format(num_embeddings,
                                                      embedding_dim))
        words_vectors = (_parse_line(line) for line in ft_file)
        int_to_vec = {vocab[word]: vec for word, vec in words_vectors}
    # stack them up
    embedding_array = np.array([int_to_vec[i] for i in range(num_embeddings)])

    embedding_matrix = tf.Variable(
        embedding_array, name='embeddings', trainable=trainable)

    return embedding_matrix


def _make_embedding_matrix(embedding_dim, vocab, trainable=True):
    """make a matrix for word embeddings"""
    embedding_matrix = tf.get_variable(
        'embeddings',
        shape=[len(vocab), embedding_dim],
        trainable=trainable,
        initializer=tf.initializers.orthogonal())
    return embedding_matrix


def load_word_embeddings(ft_vecfile, vocab, trainable=False,
                         embedding_dim=128):
    """either load pre-trained embeddings or just make a matrix"""
    if ft_vecfile:
        return _load_fasttext_embeddings(ft_vecfile, vocab, trainable)
    return _make_embedding_matrix(embedding_dim, vocab, trainable)


def _split_line(line):
    """split a line and append a newline, unless it's empty"""
    line = (token for token in line.rstrip().split(' ') if token != '')
    line = (' @@' if token == '@@' else token for token in line)
    return list(line) + ['\n']


def _load_text(path, vocab):
    """
    Load the data as a big numpy array of ints.
    Tokenising simply splits on space, although newlines are added back
    in.
    """
    with open(path) as text_file:
        tokens = [
            vocab[token] for line in text_file for token in _split_line(line)
        ]
        print('{} tokens in data'.format(len(tokens)))
        return np.array(tokens, dtype=np.int32)


def _regenerate_indices(data_length, batch_size):
    """
    generate a batch of indices, in the first quarter of the data
    """
    return np.random.randint(data_length // 4, size=(batch_size, ))


class TextDataset(object):
    """Class holding a bit of state for the dataset"""

    def __init__(self, flat_data, sequence_length, batch_size):
        """Set it up"""
        self._flat_data = flat_data
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self._batch_indices = _regenerate_indices(self._flat_data.shape[0],
                                                  self._batch_size)

    def get_batch(self, infinite=False):
        """Get some chunks of data"""
        data = np.array([
            self._flat_data[index:index + self._sequence_length]
            for index in self._batch_indices
        ])
        self._batch_indices += self._sequence_length
        if np.any(self._batch_indices > self._flat_data.shape[0]):
            if infinite:
                self._batch_indices = _regenerate_indices(
                    self._flat_data.shape[0], self._batch_size)
            else:
                raise StopIteration()
        return data

    def __iter__(self):
        """go through the batches once"""
        while True:
            yield self.get_batch(False)


def load_dataset(filepath, vocab, sequence_length, batch_size):
    """
    Get a tensor which will roll through batches of data.

    Each batch item goes through the text in order from a random
    starting position until one of them reaches the end.
    """
    # load the data into a big numpy array
    print('loading from {}'.format(filepath))
    flat_data = _load_text(filepath, vocab)

    data = tf.data.Dataset.from_generator(
        lambda: TextDataset(flat_data, sequence_length, batch_size), tf.int32,
        [batch_size, sequence_length])

    return data.make_one_shot_iterator().get_next()


def _subword_join(a, b):
    """vocab aware join operation"""
    if a.endswith('@@'):
        a = a[:-2]
        join = ''
    elif a == '\n':
        join = ''
    else:
        join = ' '
    return a + join + b


def invert_vocab(vocab):
    """Invert a vocabulary dictionary."""
    return {b: a for a, b in vocab.items()}


def to_human_readable(items, vocab, invert_vocabulary=False):
    """turn some integer labels into a words"""
    if len(items.shape) == 1:
        items = np.reshape(items, (1, -1))
    if invert_vocabulary:
        vocab = invert_vocab(vocab)
    return [
        reduce(_subword_join, [vocab[index] for index in row]) for row in items
    ]
