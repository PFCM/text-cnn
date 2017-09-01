"""
Hierarchical autoregressive convolutional neural network.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim


def generation_net(inputs):
    """
    A network to generate something conditioned purely on a single set of
    inputs (which can be replaced by its own outputs at generation time,
    although this could be fairly inefficient.
    """
    pass


def _causal_convolution(inputs, filter_size, num_channels, rate=1,
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
        result = tf.layers.conv1d(padded,
                                  num_channels,
                                  filter_size,
                                  strides=1,
                                  padding='VALID',
                                  dilation_rate=rate,
                                  name=name)
        return result


def _block(inputs, output_channels, filter_size=2,
           depth=4, dilation_base=4, name='res_block'):
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
        name (str): name for the block.

    Returns:
        tensor: `[batch, time, output_channels]` processed tensor.
    """
    with tf.variable_scope(name):
        net = inputs
        for layer in range(depth):
            net = _causal_convolution(net, filter_size, output_channels)


def upsampling_net(inputs, name='upsampler'):
    """
    A network to upsample that generates two outputs for each individual
    input. These outputs are conditioned on a receptive field extending only
    backwards in time to enable linear time generation. To achieve this we
    use causal 1D convolutions, which just amounts to padding the front of the
    input and doing a bit of a shift.

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
        # to upsample we'll just generate a couple of extra channels and then
        # reshape
        pass