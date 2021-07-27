"""
Implements HR-Net from https://arxiv.org/abs/1904.04514v1
"""
from functools import partial
from typing import Callable, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate

from . import layers
# import layers
from .utils import get_regularizer, select_final_activation
# from utils import get_regularizer, select_final_activation

import numpy as np


def convolution_block(
        x: tf.Tensor,
        conv: Callable,
        n_filter: int,
        res_connect: bool = False,
        strides: int = 1,
        n_conv: int = 2
) -> tf.Tensor:

    for _ in range(n_conv):
        conv_x = conv(x, n_filter=n_filter, stride=strides)

    return conv_x


def downsample(
        x: tf.Tensor,
        n_down: int,
        conv: Callable,
        strides: int = 2
) -> tf.Tensor:

    down_scale = x
    # n_filters = tf.shape(x).numpy()[-1]
    n_filters = x.shape[-1]
    multi_scale = []
    for _ in range(n_down):
        n_filters = n_filters * 2
        down_scale = conv(down_scale, n_filter=n_filters, stride=strides)
        multi_scale.append(down_scale)

    return multi_scale


def upsample(
        x: tf.Tensor,
        n_up: int,
        conv: Callable,
        size: tuple = (2, 2),
        interpolation: str = 'bilinear'
) -> tf.Tensor:

    up_scale = x
    # n_filters = tf.shape(x).numpy()[-1]
    n_filters = x.shape[-1]
    multi_scale = []
    for _ in range(n_up):
        n_filters = n_filters // 2
        up_scale = tf.keras.layers.UpSampling2D(size=size, interpolation=interpolation)(up_scale)
        up_scale = conv(up_scale, n_filter=n_filters, stride=1)
        multi_scale.append(up_scale)

    return multi_scale


def final_upsample(
        x: tf.Tensor,
        n_times: int,
        interpolation: str = 'bilinear'
) -> tf.Tensor:

    size = (n_times * 2, n_times *2)
    up_scale = tf.keras.layers.UpSampling2D(size=size, interpolation=interpolation)(x)

    return up_scale


def add_layer(
        list_x: list
) -> tf.Tensor:
    
    add = tf.math.add_n(list_x)

    return add


def HRNet(
    input_tensor: tf.Tensor,
    out_channels: int,
    loss: str,
    n_filter=(16, 32, 64, 128),
    n_convolutions=(2, 2, 3, 3, 3),
    kernel_dims=3,
    batch_normalization=True,
    use_bias=False,
    drop_out=(False, 0.2),
    regularize=(True, "L2", 0.001),
    padding="SAME",
    activation="relu",
    dilation_rate=1,
    cross_hair=False,
    name="HRNet",
    **kwargs
) -> tf.keras.Model:

    if loss == 'COS':
        l2_norm = True
    else:
        l2_norm = False
    regularizer = get_regularizer(*regularize)
    conv = partial(
        layers.convolutional,
        kernel_dims=kernel_dims,
        batch_normalization=batch_normalization,
        drop_out=drop_out,
        use_bias=use_bias,
        regularizer=regularizer,
        padding=padding,
        act_func=activation,
        dilation_rate=dilation_rate,
        cross_hair=cross_hair,
    )

    conv_block = partial(
        convolution_block,
        conv=conv,
        n_conv=2
    )
    down = partial(
        downsample,
        conv=conv
    )
    up = partial(
        upsample,
        conv=conv
    )

    # stage 1 (Nomenclature: e.g. 1_2=first depth 2nd block)
    x1_1 = conv_block(input_tensor, conv=conv, n_filter=n_filter[0])
    down_x1_1 = down(x1_1, n_down=1)

    x1_1 = conv(x1_1, n_filter=n_filter[0], stride=1)

    # stage 2
    x1_2 = conv_block(x1_1, n_filter=n_filter[0])
    down_x1_2 = down(x1_2, n_down=2)
    x2_1 = conv_block(down_x1_1[0], n_filter=n_filter[1])
    down_x2_1 = down(x2_1, n_down=1)
    up_x2_1 = up(x2_1, n_up=1)

    x1_2 = conv(x1_2, n_filter=n_filter[0], stride=1)
    x1_2 = add_layer([x1_2, up_x2_1[0]])
    x2_1 = conv(x2_1, n_filter=n_filter[1], stride=1)
    x2_1 = add_layer([x2_1, down_x1_2[0]])

    down_x2_1 = add_layer([down_x2_1[0], down_x1_2[1]])

    # stage 3
    x1_3 = conv_block(x1_2, n_filter=n_filter[0])
    down_x1_3 = down(x1_3, n_down=3)
    x2_2 = conv_block(x2_1, n_filter=n_filter[1])
    down_x2_2 = down(x2_2, n_down=2)
    up_x2_2 = up(x2_2, n_up=1)
    x3_1 = conv_block(down_x2_1, n_filter=n_filter[2])
    down_x3_1 = down(x3_1, n_down=1)
    up_x3_1 = up(x3_1, n_up=2)

    x1_3 = conv(x1_3, n_filter=n_filter[0], stride=1)
    x1_3 = add_layer([x1_3, up_x2_2[0], up_x3_1[1]])
    x2_2 = conv(x2_2, n_filter=n_filter[1], stride=1)
    x2_2 = add_layer([down_x1_3[0], x2_2, up_x3_1[0]])
    x3_1 = conv(x3_1, n_filter=n_filter[2], stride=1)
    x3_1 = add_layer([down_x1_3[1], down_x2_2[0], x3_1])

    down_x3_1 = add_layer([down_x1_3[2], down_x2_2[1], down_x3_1[0]])

    # stage 4
    x1_4 = conv_block(x1_3, n_filter=n_filter[0])
    down_x1_4 = down(x1_4, n_down=3)
    x2_3 = conv_block(x2_2, n_filter=n_filter[1])
    down_x2_3 = down(x2_3, n_down=2)
    up_x2_3 = up(x2_3, n_up=1)
    x3_2 = conv_block(x3_1, n_filter=n_filter[2])
    down_x3_2 = down(x3_2, n_down=1)
    up_x3_2 = up(x3_2, n_up=2)
    x4_1 = conv_block(down_x3_1, n_filter=n_filter[3])
    up_x4_1 = up(x4_1, n_up=3)

    x1_4 = conv(x1_4, n_filter=n_filter[0], stride=1)
    x1_4 = add_layer([x1_4, up_x2_3[0], up_x3_2[1], up_x4_1[2]])
    x2_3 = conv(x2_3, n_filter=n_filter[1], stride=1)
    x2_3 = add_layer([down_x1_4[0], x2_3, up_x3_2[0], up_x4_1[1]])
    x3_2 = conv(x3_2, n_filter=n_filter[2], stride=1)
    x3_2 = add_layer([down_x1_4[1], down_x2_3[0], x3_2, up_x4_1[0]])
    x4_1 = conv(x4_1, n_filter=n_filter[3], stride=1)
    x4_1 = add_layer([down_x1_4[2], down_x2_3[1], down_x3_2[0], x4_1])

    # final block upsample & concat
    x2_3 = final_upsample(x2_3, n_times=1)
    x3_2 = final_upsample(x3_2, n_times=2)
    x4_1 = final_upsample(x4_1, n_times=4)
    final_block = Concatenate()([x1_4, x2_3, x3_2, x4_1])

    # final output layer
    logits = layers.last(
        final_block, kernel_dims=1, n_filter=out_channels, stride=1, dilation_rate=dilation_rate,
        padding=padding, act_func=select_final_activation(loss, out_channels), use_bias=False,
        regularizer=regularizer, l2_normalize=l2_norm,
    )

    return tf.keras.Model(inputs=input_tensor, outputs=logits)


# a = np.ones((1, 96, 96, 1))
# input = tf.convert_to_tensor(a)
# input = tf.ones(shape=(1, 96, 96, 1))
# model_name = 'hrnet.png'
# model = HRNet(input_tensor=input, out_channels=2, loss='CEL')
# model.compile(
#             loss=tf.keras.losses.CategoricalCrossentropy(),
#             metrics="acc",
#             optimizer=tf.keras.optimizers.Adam(),
#         )
# tf.keras.utils.plot_model(model, to_file=model_name, show_shapes=True)









