"""
Train a model, either to generate or upsample.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

import textcnn.models as model
import textcnn.read as read

tf.app.flags.DEFINE_integer(
    'rate', 1, 'downsampling applied to the data. If it is 1, we '
    'learn the basic generation model. If > 1 we '
    'learn a model that takes data downsampled by '
    '`rate` and upsamples it by a factor of two.')
tf.app.flags.DEFINE_string('logdir', '/tmp/abcdefg',
                           'where to store logfiles and checkpoints.')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'step size for sgd')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size for sgd')
tf.app.flags.DEFINE_integer(
    'sequence_length', 250, 'sequence length. In '
    'practice it has to be the maximium of this and '
    'the receptive field of the model')
tf.app.flags.DEFINE_string(
    'data_path', None, 'The data. Should be a single text file, '
    'preprocessed with subword-nmt')
tf.app.flags.DEFINE_string(
    'vocab_path', None, 'path to the vocabulary as generated by '
    'subword-nmt')
tf.app.flags.DEFINE_string('embedding_path', None,
                           'path to the embeddings generated by fastText')
FLAGS = tf.app.flags.FLAGS


def _downsample_inputs(data, downsample):
    """reduce in time by averaging"""
    data = tf.expand_dims(data, 1)
    data = tf.nn.avg_pool(data, [1, 1, downsample, 1], [1, 1, downsample, 1],
                          'SAME')
    return data


def get_data(data_path, vocab_path, embedding_path, downsampling,
             sequence_length, batch_size):
    """get data, embed it and potentially downsample it"""
    with tf.variable_scope('data'):
        vocab = read.load_vocab(vocab_path)
        print('{} items in vocab'.format(len(vocab)))
        data_tensor = read.load_dataset(data_path, vocab, sequence_length,
                                        batch_size)
    with tf.variable_scope('embeddings'):
        embedding_matrix = read.load_word_embeddings(embedding_path, vocab)
        embedded_data = tf.nn.embedding_lookup(embedding_matrix, data_tensor)
        if downsampling > 1:
            embedded_data = _downsample_inputs(embedded_data, downsample)
    return embedded_data, vocab


def get_loss(targets, net_outputs):
    """Loss is just the squared error between the inputs to the network and its
    outputs. To make it actually a prediction task the first target is ignored.
    Targets should therefore have one more timestep than the net_outputs."""
    with tf.variable_scope('loss'):
        return tf.reduce_mean(
            tf.reduce_sum(tf.square(targets[:, 1:, :] - net_outputs), axis=1))


def get_train_step(loss):
    """Get a training step to update `loss`."""
    with tf.variable_scope('training'):
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        return opt.minimize(loss)


def main(_):
    """actually do a things"""
    target_batch, vocab = get_data(FLAGS.data_path, FLAGS.vocab_path,
                                   FLAGS.embedding_path, FLAGS.rate // 2,
                                   FLAGS.sequence_length, FLAGS.batch_size)
    print('\n'.join(
        '{}: {}'.format(a.split('@@')[0], b) for a, b in vocab.items()))
    raise SystemExit
    if FLAGS.rate > 1:
        input_batch = _downsample_inputs(target_batch, 2)
    else:
        input_batch = target_batch

    # TODO: the different kinds of net (generator and upsampler)
    # should we just do a nearest neighbour lookup in the embeddings for
    # the last layer?

    net_out = model.training_generator_net(input_batch[:, :-1, :])
    loss = get_loss(target_batch, net_out)
    print('model ready, getting training graph')
    train_step = get_train_step(loss)
    print('ready to train')

    with tf.Session() as sess:
        print('initialising...', end='', flush=True)
        sess.run(tf.global_variables_initializer())
        print('\rinitialised       ')

        step = 0
        while True:
            try:
                loss_val = sess.run(loss)
                print(
                    '\r{}: {}      '.format(step, loss_val),
                    end='',
                    flush=True)
                step += 1
            except KeyboardInterrupt:
                print('keyboard interrupt, exiting')
                break


if __name__ == '__main__':
    tf.app.run()