"""train a baseline RNN"""

import argparse

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
        '--embedding_path', help='path to pre-trained embeddings')

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
        embedding_matrix = data.load_word_embeddings(embedding_path, vocab)

    return data_tensor, embedding_matrix, vocab


def get_loss(targets, predictions):
    """get the loss"""
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=targets, logits=predictions)
    tf.summary.scalar('loss', loss)
    return loss


def main(args=None):
    """run the thing"""
    args = _parse_args(args)
    data_batch, embedding_matrix, vocab = get_data(
        args.data_path, args.vocab_path, args.embedding_path,
        args.sequence_length + 1, args.batch_size)
    input_batch = data_batch[:, :-1]
    target_batch = data_batch[:, 1:]

    preds = rnn_model(input_batch, [1024], embedding_matrix)
    check = tf.argmax(preds, -1, output_type=tf.int32)
    print('got model')

    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(0.001)
    loss = get_loss(target_batch, preds)
    train_step = opt.minimize(loss, global_step=global_step)
    print('got training ops')

    hooks = [
        tf.train.CheckpointSaverHook(args.logdir, save_secs=120),
        tf.train.SummarySaverHook(
            save_secs=30,
            output_dir=args.logdir,
            summary_op=tf.summary.merge_all())
    ]

    with tf.train.SingularMonitoredSession(hooks=hooks) as sess:
        print('ready to train')
        while not sess.should_stop():
            loss_val, _, step = sess.run([loss, train_step, global_step])
            if (step % 10) == 0:
                print('\r{}: {:.4f}'.format(step, loss_val))
            if (step % 100) == 0:
                # just a check on the auto-encoding, not a real sample
                sample = sess.run(check)
                sample = data.to_human_readable(sample, vocab, True)
                print()
                print('\n~~~~~~~~~\n'.join(sample))


if __name__ == '__main__':
    main()
