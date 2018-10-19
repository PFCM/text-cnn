"""
Hierarchical autoregressive convolutional neural network.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def training_generator_net(inputs, num_blocks=2, block_size=3, block_width=64):
    """
    A network to generate something conditioned purely on a single set of
    input embeddings.

    The training is fairly straightforward, the idea being that it will be
    able to be applied autoregressively to generate new things
    unconditionally. This will definitely be pretty slow, but hopefully
    the sequences will be pretty short.

    Args:
        inputs: [batch_size, time, embedding_size] float tensor of inputs.
        vocab_size: vocabulary size, the size of the final outputs.
        num_blocks: number of residual blocks in each stage of the network.
            Defaults to 2, for a fairly small test network.
        block_size: number of individual residual units in each block.
            Defaults to 3.
        block_width: number of channels in each of the residual blocks,
            also the width of the hidden codes. Defaults to 128.

    Returns:
        outputs: same shape as the inputs, it's up to the user to slice and
            for training.
    """
    with tf.variable_scope('generator'):
        input_width = inputs.get_shape()[-1].value
        net = _block(
            inputs,
            block_width,
            filter_size=2,
            depth=block_size,
            name='block_0')
        for block in range(num_blocks - 1):
            net += _block(
                net,
                block_width,
                filter_size=2,
                depth=block_size,
                name='block_{}'.format(block + 1))
        net = tf.layers.conv1d(net, input_width, 1, 1)

    return net


def _causal_convolution(inputs,
                        filter_size,
                        num_channels,
                        rate=1,
                        name='causal_conv'):
    """Linear convolution in which each output is dependent only on preceding
    inputs.

    There are two ways to implement this:
        - as a convolution with masked filters of twice the size.
        - as a convolution of the desired size on a padded input.

    The latter should be the most appropriate.

    Args:
        inputs: tensor with shape `[batch, time, channels]` to be convolved.
        filter_size (int): the number taps in each filter.
        num_channels (int): the number of output channels.
        rate (int): dilation of the filters. Defaults to 1 for no dilation.

    Returns:
        tensor: shape `[batch, time, num_channels]`, tensor with new number of
            channels.
    """
    with tf.name_scope(name):
        # figure out how much padding is required
        padding = (filter_size - 1) * rate
        padded = tf.pad(inputs, [[0, 0], [padding, 0], [0, 0]])
        # now if we just do a valid convolution, everything should work out
        result = tf.layers.conv1d(
            padded,
            num_channels,
            filter_size,
            strides=1,
            padding='VALID',
            dilation_rate=rate,
            name=name)
        return result


def _block(inputs,
           output_channels,
           filter_size=4,
           depth=4,
           dilation_base=2,
           causal=True,
           name='res_block'):
    """One block which we use in between residual layers. Consists of a number
    of stacked dilated convolutions and layer normalisation.

    Args:
        inputs (tensor): `[batch, time, channels]` input tensor.
        output_channels (int): number of output channels. The very first
            convolution will be responsible for changing the number of channels
            if this is different to the number of input channels -- all the
            others will then have the same amount.
        depth (int): number of stacked convolutions. Defaults to 4.
        dilation_base (int): number controlling the growth of the receptive
            field. Each layer, starting at 0, will have dilation rate
            `dilation_base ** layer`.
        causal (bool): if True, only causal convolutions are used. Otherwise
            normal convolutions.
        name (str): name for the block.

    Returns:
        tensor: `[batch, time, output_channels]` processed tensor.
    """
    with tf.variable_scope(name):
        net = inputs
        for layer in range(depth):
            if causal:
                net = _causal_convolution(
                    net,
                    filter_size,
                    output_channels,
                    rate=dilation_base**layer,
                    name='conv_{}'.format(layer))
            else:
                net = tf.layers.conv1d(
                    net,
                    output_channels,
                    filter_size,
                    dilation_rate=dilation_base**layer,
                    strides=1,
                    padding='SAME',
                    name='conv_{}'.format(layer))
            net = layer_norm(net, name='ln_{}'.format(layer))
            net = tf.nn.relu(net)
        return net


def layer_norm(inputs, epsilon=1e-8, name='layer_norm'):
    """
    Layer normalisation along the last axis.
    """
    with tf.variable_scope(name):
        shape = inputs.get_shape()
        beta = tf.get_variable(
            'beta', [shape[-1].value],
            initializer=tf.constant_initializer(0),
            trainable=True)
        gamma = tf.get_variable(
            'gamma', [shape[-1].value],
            initializer=tf.constant_initializer(1),
            trainable=True)

        mean, variance = tf.nn.moments(
            inputs, axes=[len(shape) - 1], keep_dims=True)

        result = (inputs - mean) / tf.sqrt(variance + epsilon)
        result = tf.cast(result, tf.float32)  # probably could be earlier

        return gamma * result + beta


def upsampling_net(inputs,
                   num_blocks=3,
                   block_depth=3,
                   block_width=128,
                   causal=True,
                   name='upsampler'):
    """
    A network to upsample that generates two outputs for each individual
    input. These outputs can be conditioned on a receptive field extending only
    backwards in time to enable continuous generation in linear time. To
    achieve this we use causal 1D convolutions, which just amounts to padding
    the front of the input and doing a bit of a shift. If this is not desired
    we can just use normal convolutions which will likely perform slightly
    better but means we have to generate the entire lower level representation
    first.

    The input size (and the number of filters throughout) is just the same as
    the number of input channels.

    Args:
        input (tensor): a `[batch, time, channels]` tensor which represents
            some kind of signal.
        name (Optional[str]): a name for the scopes under which we should add
            all the variables.

    Returns:
        tensor: a `[batch, time*2, channels]` upsampled tensor.
    """
    with tf.variable_scope(name):
        # to upsample we will resample the original signal and then push it
        # through a few layers
        expanded_inputs = tf.expand_dims(inputs, 1)
        new_shape = expanded_inputs.get_shape().as_list()[1:3]
        new_shape[1] *= 2
        expanded_inputs = tf.image.resize_nearest_neighbor(
            expanded_inputs, size=new_shape)
        net = tf.squeeze(expanded_inputs, 1)

        for block in range(num_blocks):
            net += _block(
                net,
                block_width,
                filter_size=2,
                depth=block_depth,
                causal=causal,
                name='block_{}'.format(block))

        return net
