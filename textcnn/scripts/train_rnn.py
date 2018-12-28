"""train a baseline RNN"""

import argparse
import time
from collections import deque

import tensorflow as tf

import textcnn.read as data
from textcnn.models.rnn import rnn_model


def _parse_args(args=None):
    """get command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument(
        '--logdir', default='/tmp/rnn', help='where to store logs')
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=256,
        help='max sequence length to train on')
    parser.add_argument('--data_path', help='path to the text data')
    parser.add_argument('--vocab_path', help='path to the vocab file')
    parser.add_argument(
        '--embedding_path',
        default=None,
        help='path to pre-trained embeddings')
    parser.add_argument(
        '--max_time', type=float, help='maximum time to train for in seconds')

    return parser.parse_args(args)


def get_data(data_path, vocab_path, embedding_path, sequence_length,
             batch_size):
    """get data, embed it and potentially downsample it"""
    with tf.variable_scope('data'):
        vocab = data.load_vocab(vocab_path)
        print('{} items in vocab'.format(len(vocab)))
        data_tensor = data.load_dataset(data_path, vocab, sequence_length,
                                        batch_size)
    with tf.variable_scope('embeddings'):
        embedding_matrix = data.load_word_embeddings(
            embedding_path, vocab, trainable=True)

    return data_tensor, embedding_matrix, vocab


def get_loss(targets, predictions):
    """get the loss"""
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=targets, logits=predictions)
    tf.summary.scalar('loss', loss)
    return loss


def _should_stop(start, max_time):
    """see if it's gone for too long"""
    return (time.time() - start) >= max_time


def main(args=None):
    """run the thing"""
    args = _parse_args(args)
    data_batch, embedding_matrix, vocab = get_data(
        args.data_path, args.vocab_path, args.embedding_path,
        args.sequence_length + 1, args.batch_size)
    input_batch = data_batch[:, :-1]
    target_batch = data_batch[:, 1:]

    preds = rnn_model(input_batch, [512], embedding_matrix)
    check = tf.argmax(preds, -1, output_type=tf.int32)
    print('got model')

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.Variable(0.001, trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate)
    cut_lr = tf.assign_sub(learning_rate, learning_rate / 2)
    tf.summary.scalar('learning_rate', learning_rate)
    loss = get_loss(target_batch, preds)
    train_step = opt.minimize(loss, global_step=global_step)
    print('got training ops')

    hooks = [
        tf.train.CheckpointSaverHook(args.logdir, save_secs=60),
        tf.train.SummarySaverHook(
            save_secs=30,
            output_dir=args.logdir,
            summary_op=tf.summary.merge_all())
    ]

    with tf.train.SingularMonitoredSession(hooks=hooks) as sess:
        print('ready to train')
        start_time = time.time()
        # rough early stopping
        loss_avg = None
        loss_vals = deque(maxlen=25)
        while not sess.should_stop() and not _should_stop(
                start_time, args.max_time):
            loss_val, _, step = sess.run([loss, train_step, global_step])
            if loss_avg is None:
                loss_avg = loss_val
            else:
                loss_avg = (0.99 * loss_avg) + (0.01 * loss_val)
            loss_vals.append(loss_val)
            if len(loss_vals) == 25 and all(
                    map(lambda x: x > loss_avg, loss_vals)):
                print('not looking good, dropping learning rate')
                sess.run(cut_lr)
            if (step % 10) == 0:
                print('\r{}: {:.4f}'.format(step, loss_val))
            if (step % 100) == 0:
                # just a check on the auto-encoding, not a real sample
                sample = sess.run(check)
                sample = data.to_human_readable(sample[:2, ...], vocab, True)
                print()
                print('\n~~~~~~~~~\n'.join(sample))
        print('final loss average: {}'.format(loss_avg))


if __name__ == '__main__':
    main()
