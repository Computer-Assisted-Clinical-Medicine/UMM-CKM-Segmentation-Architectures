"""
Implements multiple different kinds of UNets
"""
from functools import partial

import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate

from . import layers
from .utils import get_regularizer, select_final_activation


def unet(input_tensor: tf.Tensor, out_channels: int, loss: str, n_filter=(8, 16, 32, 64, 128),
         filter_shape=3, stride=1, batch_normalization=True, use_bias=False, drop_out=(False, 0.2),
         upscale='TRANS_CONV', downscale='MAX_POOL', regularize=(True, "L2", 0.001), padding='SAME', activation='relu',
         name='Unet', ratio=1, dilation_rate=1, cross_hair=False, **kwargs):
    """
    Implements U-Net (https://arxiv.org/abs/1505.04597) as the backbone. The add-on architectures are Attention U-Net
    (https://arxiv.org/abs/1804.03999), CBAMUnet, CBAMAttnUnet, SEUnet and SEAttnUnet. Where Convolutional block
    attention modules (CBAM - https://arxiv.org/abs/1807.06521) and Squeeze & excitation blocks
    (SE - https://arxiv.org/abs/1709.01507) are added to the encoder of U-Net or Attention U-Net to obtain CBAM and SE
    attention U-Nets respectively.

    :param input_tensor: input tensorflow tensor/image.
    :param out_channels: number of classes that needs to be segmented.
    :param loss: loss function as a string
    :param n_filter: a list containing number of filters for conv layers (encoder block: 1 to 5, decoder block: 4 to 1)
    By default: [8, 16, 32, 64, 128].
    :param filter_shape: shape of all the convolution filter, by default: 3.
    :param stride: stride for all the conv layers, by default: 1.
    :param batch_normalization: boolean value, whether to apply batch_norm or not. By default: True.
    :param use_bias: boolean value, whether to apply bias or not. If batch_normalization is true then use_bias must be
    false and vice versa By default: False.
    :param drop_out: a list containing a boolean, whether to apply dropout to conv layers or not. The number signifies
    the probability of dropout. By default: [False, 0.2].
    :param upscale: The strategy to use for upscaling features. By default: 'TRANS_CONV'.
    :param downscale: The strategy to downscale features. Options: 'MAX_POOL', 'STRIDE'. By default: 'MAX_POOL'.
    :param regularize: The value for l2 regularization. By default: 0.00001.
    :param padding: The strategy to pad the features. By default: 'SAME'.
    :param activation: The activation used after each layer. By default: 'relu'.
    :param name: The network that the user wants to implement. Must be one of the following: 'Unet', 'SEUnet',
     'SEAttnUnet', 'CBAMUnet', 'CBAMAttnUnet', 'AttnUnet'. By default: Unet.
    :param se_layer: boolean, whether to use se block or not.
    :param cbam: boolean, whether to use CBAM or not.
    :param ratio: The ratio by which features are reduced in SE or CBAM channel attention.
    :param dilation_rate: dilation rate for convolutions. By default: 1.
    :param cross_hair: Boolean, whether to use cross hair convolutions or not. By default: False.

    :return: A model specified in the name argument.
    """

    available_models = ['Unet', 'SEUnet', 'SEAttnUnet',
                        'CBAMUnet', 'CBAMAttnUnet', 'AttnUnet']
    se_models = ['SEUnet', 'SEAttnUnet']
    cbam_models = ['CBAMUnet', 'CBAMAttnUnet']
    attn_models = ['AttnUnet', 'SEAttnUnet', 'CBAMAttnUnet']
    if name not in available_models:
        raise NotImplementedError(f'Architecture:{name} not implemented')
    if name not in ['Unet', 'AttnUnet']:
        if ratio == 1:
            raise ValueError('For SE or CBAM blocks to work, use ratio higher than 1')

    outputs = {}
    rank = len(input_tensor.shape) - 2
    filter_shape = [filter_shape] * rank
    stride = [stride] * rank

    regularizer = get_regularizer(*regularize)

    # set up permanent arguments of the layers
    conv = partial(layers.convolutional, filter_shape=filter_shape, stride=stride,
                   batch_normalization=batch_normalization, drop_out=drop_out,
                   use_bias=use_bias, regularizer=regularizer, padding=padding,
                   act_func=activation, dilation_rate=dilation_rate, cross_hair=cross_hair)
    downscale = partial(layers.downscale, downscale=downscale, filter_shape=filter_shape,
                        act_func=activation, stride=stride, use_bias=use_bias,
                        regularizer=regularizer, padding=padding, dilation_rate=dilation_rate,
                        cross_hair=cross_hair)  # here stride is multiplied by 2 in func to downscale by 2
    upscale = partial(layers.upscale, upscale=upscale, filter_shape=filter_shape,
                      act_func=activation, stride=stride, use_bias=use_bias,
                      regularizer=regularizer, padding=padding, dilation_rate=dilation_rate,
                      cross_hair=cross_hair)  # stride multiplied by 2 in function
    gate_signal = partial(layers.unet_gating_signal, batch_normalization=batch_normalization)
    attn_block = partial(layers.attn_gating_block, use_bias=use_bias, batch_normalization=batch_normalization)
    se_block = partial(layers.se_block, activation=activation, ratio=ratio)
    cbam_block = partial(layers.cbam_block, ratio=ratio)

    # input layer
    img_input = input_tensor
    # encoder block 1
    x1_0 = conv(img_input, n_filter=n_filter[0])
    x1_1 = conv(x1_0, n_filter=n_filter[0])
    if name in se_models:
        x1_1 = se_block(x1_1, n_filter[0])
    if name in cbam_models:
        x1_1 = cbam_block(x1_1)
    residual1 = Add()([x1_0, x1_1])  # also a skip connect for decoder
    downscale1 = downscale(residual1, n_filter=n_filter[0])

    # encoder block 2
    x2_0 = conv(downscale1, n_filter=n_filter[1])
    x2_1 = conv(x2_0, n_filter=n_filter[1])
    if name in se_models:
        x2_1 = se_block(x2_1, n_filter[1])
    if name in cbam_models:
        x2_1 = cbam_block(x2_1)
    residual2 = Add()([x2_0, x2_1])  # also a skip connect for decoder
    downscale2 = downscale(residual2, n_filter=n_filter[1])

    # encoder block 3
    x3_0 = conv(downscale2, n_filter=n_filter[2])
    x3_1 = conv(x3_0, n_filter=n_filter[2])
    x3_2 = conv(x3_1, n_filter=n_filter[2])
    if name in se_models:
        x3_2 = se_block(x3_2, n_filter[2])
    if name in cbam_models:
        x3_2 = cbam_block(x3_2)
    residual3 = Add()([x3_0, x3_2])  # also a skip connect for decoder
    downscale3 = downscale(residual3, n_filter=n_filter[2])

    # encoder block 4
    x4_0 = conv(downscale3, n_filter=n_filter[3])
    x4_1 = conv(x4_0, n_filter=n_filter[3])
    x4_2 = conv(x4_1, n_filter=n_filter[3])
    if name in se_models:
        x4_2 = se_block(x4_2, n_filter[3])
    if name in cbam_models:
        x4_2 = cbam_block(x4_2)
    residual4 = Add()([x4_0, x4_2])  # also a skip connect for decoder
    downscale4 = downscale(residual4, n_filter=n_filter[3])

    # bottom block 5
    x5_0 = conv(downscale4, n_filter=n_filter[4])
    x5_1 = conv(x5_0, n_filter=n_filter[4])
    x5_2 = conv(x5_1, n_filter=n_filter[4])
    residual5 = Add()([x5_0, x5_2])  # also a skip connect for decoder

    upscale5 = upscale(residual5, n_filter=n_filter[3])
    # decoder block 6
    if name in attn_models:
        gate6 = tf.identity(residual5)
        gate6 = gate_signal(gate6)
        attn6 = attn_block(residual4, gate6, n_filter[3])
        concat6 = Concatenate()([upscale5, attn6])
    else:
        concat6 = Concatenate()([upscale5, residual4])
    x6_0 = conv(concat6, n_filter=n_filter[3])
    x6_1 = conv(x6_0, n_filter=n_filter[3])
    x6_2 = conv(x6_1, n_filter=n_filter[3])
    residual6 = Add()([x6_0, x6_2])

    upscale6 = upscale(residual6, n_filter=n_filter[2])

    # decoder block 7
    if name in attn_models:
        gate7 = tf.identity(residual6)
        gate7 = gate_signal(gate7)
        attn7 = attn_block(residual3, gate7, n_filter[2])
        concat7 = Concatenate()([upscale6, attn7])
    else:
        concat7 = Concatenate()([upscale6, residual3])
    x7_0 = conv(concat7, n_filter=n_filter[2])
    x7_1 = conv(x7_0, n_filter=n_filter[2])
    x7_2 = conv(x7_1, n_filter=n_filter[2])
    residual7 = Add()([x7_0, x7_2])

    upscale7 = upscale(residual7, n_filter=n_filter[1])

    # decoder block 8
    if name in attn_models:
        gate8 = tf.identity(residual7)
        gate8 = gate_signal(gate8)
        attn8 = attn_block(residual2, gate8, n_filter[1])
        concat8 = Concatenate()([upscale7, attn8])
    else:
        concat8 = Concatenate()([upscale7, residual2])
    x8_0 = conv(concat8, n_filter=n_filter[1])
    x8_1 = conv(x8_0, n_filter=n_filter[1])
    residual8 = Add()([x8_0, x8_1])

    upscale8 = upscale(residual8, n_filter=n_filter[0])

    # decoder block 9
    if name in attn_models:
        gate9 = tf.identity(residual8)
        gate9 = gate_signal(gate9)
        attn9 = attn_block(residual1, gate9, n_filter[0])
        concat9 = Concatenate()([upscale8, attn9])
    else:
        concat9 = Concatenate()([upscale8, residual1])
    x9_0 = conv(concat9, n_filter=n_filter[0])
    x9_1 = conv(x9_0, n_filter=n_filter[0])
    residual9 = Add()([x9_0, x9_1])

    # final output layer
    logits = layers.last(residual9, outputs, filter_shape=1, n_filter=out_channels,
                         stride=stride, dilation_rate=dilation_rate, padding=padding,
                         act_func=select_final_activation(loss, out_channels),
                         use_bias=False, regularizer=regularizer, l2_normalize=False)

    return tf.keras.Model(inputs=input_tensor, outputs=logits)
