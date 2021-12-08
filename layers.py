"""This module contains multiple layers, which are mostly used the UNets.
"""
# pylint: disable=invalid-name
import logging
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Lambda,
    Permute,
    Reshape,
    add,
    multiply,
)

# configure logger
logger = logging.getLogger(__name__)


def activation(act_func: str) -> tf.keras.layers.Layer:
    """This function takes the name of the activation function and returns the
    corresponding Layer

    Parameters
    ----------
    act_func : str
        The function name it can be:
        - a keras activation (like relu, elu)
        - swish
        - elu

    Returns
    -------
    layer
        The layer
    """
    if act_func in tf.keras.activations.__dict__:
        return tf.keras.layers.Activation(act_func)
    elif act_func == "leaky_relu":
        return tf.keras.layers.LeakyReLU()
    elif act_func == "swish":
        return swish
    elif act_func == "elu":
        return tf.keras.layers.ELU()
    else:
        raise ValueError(f"Activation function {act_func} unknown.")


def swish(x, beta=1) -> tf.keras.layers.Layer:
    return tf.keras.layers.Multiply(x, tf.keras.activations.sigmoid(beta * x))


def expend_as(tensor: tf.Tensor, rep: int, axis: int, name=None) -> tf.keras.layers.Layer:
    return Lambda(
        lambda x, repnum: K.repeat_elements(x, repnum, axis=axis),
        arguments={"repnum": rep},
        name=name,
    )(tensor)


def attn_gating_block(
    x: tf.Tensor,
    gate: tf.Tensor,
    inter_shape: int,
    use_bias: bool,
    batch_normalization: bool,
    name=None,
    **kwargs,
) -> tf.Tensor:
    """take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same feature channels (theta_x)
    then, upsample g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients"""

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gate)

    if tf.rank(x).numpy() == 4:
        theta_x = tf.keras.layers.Conv2D(
            inter_shape, (2, 2), strides=(2, 2), padding="same", use_bias=use_bias
        )(x)
        shape_theta_x = K.int_shape(theta_x)
        phi_g = tf.keras.layers.Conv2D(inter_shape, (1, 1), padding="same")(gate)
        upsample_g = tf.keras.layers.Conv2DTranspose(
            inter_shape,
            (3, 3),
            strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
            padding="same",
            use_bias=use_bias,
        )(phi_g)
        concat_xg = add([upsample_g, theta_x])
        act_xg = Activation("relu")(concat_xg)
        psi = tf.keras.layers.Conv2D(1, (1, 1), padding="same", use_bias=use_bias)(act_xg)
        sigmoid_xg = Activation("sigmoid")(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = tf.keras.layers.UpSampling2D(
            size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2])
        )(sigmoid_xg)
        upsample_psi = expend_as(upsample_psi, shape_x[3], axis=3)
        y = multiply([upsample_psi, x])
        result = tf.keras.layers.Conv2D(shape_x[3], (1, 1), padding="same")(y)

    elif tf.rank(x).numpy() == 5:
        theta_x = tf.keras.layers.Conv3D(
            inter_shape, (2, 2, 2), strides=(2, 2, 2), padding="same", use_bias=use_bias
        )(x)
        shape_theta_x = K.int_shape(theta_x)
        phi_g = tf.keras.layers.Conv3D(inter_shape, (1, 1, 1), padding="same")(gate)
        upsample_g = tf.keras.layers.Conv3DTranspose(
            inter_shape,
            (3, 3, 3),
            strides=(
                shape_theta_x[1] // shape_g[1],
                shape_theta_x[2] // shape_g[2],
                shape_theta_x[3] // shape_g[3],
            ),
            padding="same",
            use_bias=use_bias,
        )(phi_g)
        concat_xg = add([upsample_g, theta_x])
        act_xg = Activation("relu")(concat_xg)
        psi = tf.keras.layers.Conv3D(1, (1, 1, 1), padding="same", use_bias=use_bias)(
            act_xg
        )
        sigmoid_xg = Activation("sigmoid")(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = tf.keras.layers.UpSampling3D(
            size=(
                shape_x[1] // shape_sigmoid[1],
                shape_x[2] // shape_sigmoid[2],
                shape_x[3] // shape_sigmoid[3],
            )
        )(sigmoid_xg)
        upsample_psi = expend_as(upsample_psi, shape_x[4], axis=4)
        y = multiply([upsample_psi, x])
        result = tf.keras.layers.Conv3D(shape_x[4], (1, 1, 1), padding="same")(y)

    if batch_normalization:
        result = tf.keras.layers.BatchNormalization()(result)

    return result


def unet_gating_signal(x: tf.Tensor, batch_normalization: bool, name=None) -> tf.Tensor:
    """this is simply 1x1 convolution, bn, activation used for calculating gating signal for attention block."""

    shape = K.int_shape(x)
    if tf.rank(x).numpy() == 4:
        x = tf.keras.layers.Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same")(x)
    elif tf.rank(x).numpy() == 5:
        x = tf.keras.layers.Conv3D(
            shape[4] * 1, (1, 1, 1), strides=(1, 1, 1), padding="same"
        )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def se_block(
    in_block: tf.Tensor, n_filter: int, act_func: str, ratio: float, **kwargs
) -> tf.Tensor:
    """Adds a channelwise attention to the layer called squeeze and excitation block.

    in_block :
        input feature map
    n_filter :
        number of desire output n_filter, preferrably same as the channels in in_block.
    ratio :
        ratio to divide the channel number for excitation.

    Returns
    -------
    tf.Tensor
        scaled input with the se matrix.
    """

    if tf.rank(in_block).numpy() == 4:
        x = tf.keras.layers.GlobalAveragePooling2D()(in_block)
        x = tf.keras.layers.Dense(n_filter // ratio, activation=act_func)(x)
        x = tf.keras.layers.Dense(n_filter, activation="sigmoid")(x)
    else:
        # x = tf.keras.layers.GlobalAveragePooling3D()(in_block)
        num_slices = in_block.shape[1]
        x = tfa.layers.AdaptiveAveragePooling3D((num_slices, 1, 1))(in_block)
        x = tf.keras.layers.Dense(n_filter // ratio, activation=act_func)(x)
        x = tf.keras.layers.Dense(n_filter, activation="sigmoid")(x)

    return multiply([in_block, x])


def cbam_block(cbam_feature: tf.Tensor, ratio=4, **kwargs) -> tf.Tensor:
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature: tf.Tensor, ratio=8) -> tf.Tensor:
    """part of cbam attention module"""

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(
        channel // ratio,
        activation="elu",  # try elu afterwards
        use_bias=False,
        bias_initializer="zeros",
    )
    shared_layer_two = Dense(
        channel,
        use_bias=False,
        bias_initializer="zeros",
    )

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation("sigmoid")(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature: tf.Tensor) -> tf.Tensor:
    """part of cbam attention module"""

    kernel_size = 5

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        use_bias=False,
    )(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def psa_channel_attention(input_feature: tf.Tensor) -> tf.Tensor:
    """Polarized self attention channel block from https://arxiv.org/pdf/2107.00782v1.pdf"""
    channels = tf.shape(input_feature).numpy()[-1]
    hw = tf.shape(input_feature).numpy()[-2] * tf.shape(input_feature).numpy()[-3]

    wq = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1)(
        input_feature
    )  # h x w x 1
    wv = tf.keras.layers.Conv2D(filters=channels // 2, kernel_size=1, strides=1)(
        input_feature
    )  # h x w x c/2

    sigma2 = tf.reshape(wq, (hw, 1, 1))  # hw x 1 x 1
    fsm = tf.keras.layers.Softmax()(sigma2)  # hw x 1 x 1
    sigma1 = tf.reshape(wv, (channels // 2, hw))  # c/2 x hw

    prod = tf.tensordot(sigma1, fsm, axes=1)  # c/2 x 1 x 1
    prod = tf.reshape(prod, (1, 1, channels // 2))  # 1 x 1 x c/2

    wz = tf.keras.layers.Conv2D(filters=channels, kernel_size=1)(prod)  # 1 x 1 x c
    layer_norm = tf.keras.layers.LayerNormalization(axis=-1)(wz)
    fsg = tf.math.sigmoid(layer_norm)  # 1 x 1 x c

    return multiply([input_feature, fsg])


def psa_spatial_attention(input_feature: tf.Tensor) -> tf.Tensor:
    """Polarized self attention spatial block from https://arxiv.org/pdf/2107.00782v1.pdf"""
    channels = tf.shape(input_feature).numpy()[-1]
    h, w = tf.shape(input_feature).numpy()[-2], tf.shape(input_feature).numpy()[-3]
    hw = tf.shape(input_feature).numpy()[-2] * tf.shape(input_feature).numpy()[-3]

    wv = tf.keras.layers.Conv2D(filters=channels // 2, kernel_size=1, strides=1)(
        input_feature
    )  # h x w x c/2
    wq = tf.keras.layers.Conv2D(filters=channels // 2, kernel_size=1, strides=1)(
        input_feature
    )  # h x w x c/2

    sigma2 = tf.reshape(wv, (channels // 2, hw))  # c/2 x hw
    fgp = GlobalAveragePooling2D()(wq)  # c/2
    sigma1 = tf.reshape(fgp, (1, channels // 2))
    fsm = tf.keras.layers.Softmax()(sigma1)  # 1 x c/2

    prod = tf.tensordot(fsm, sigma2, axes=1)  # 1 x hw
    prod = tf.reshape(prod, (h, w, 1))

    fsg = tf.math.sigmoid(prod)

    return multiply([input_feature, fsg])


def psa_attention_block(
    input_feature: tf.Tensor, strategy: str = "parallel", **kwargs
) -> tf.Tensor:
    """Polarized self attention block from https://arxiv.org/pdf/2107.00782v1.pdf"""
    if strategy == "parallel":
        ch_attn = psa_channel_attention(input_feature=input_feature)
        sp_attn = psa_spatial_attention(input_feature=input_feature)
        final_attn = add()([ch_attn, sp_attn])
    elif strategy == "sequential":
        raise NotImplementedError(
            "sequential strategy not yet implemented, please use parallel strategy"
        )
    else:
        raise ValueError("Please select strategy from ['parallel', 'sequential']")

    return final_attn


def convolutional(
    x: tf.Tensor,
    kernel_dims: Union[int, Tuple],
    n_filter: int,
    stride: Union[int, Tuple],
    padding: str,
    dilation_rate: Union[int, Tuple],
    act_func: str,
    use_bias: bool,
    batch_normalization: bool,
    drop_out: Tuple,
    regularizer: Optional[tf.keras.regularizers.Regularizer],
    cross_hair,
) -> tf.Tensor:
    """
    Implements a convolutional layer: convolution + activation

    Given an input tensor `x` of shape **TODO**, a filter kernel shape `filter_shape`,
    the number of filters `n_filter`, this function performs the following,
        - passes `x` through a convolution operation with stride [1,1] (See operation.convolution() )
        - passes `x` through an activation operation (See operation.activation() ) and return
    """

    logger.debug("Convolution")
    logger.debug("Input: %s", x.shape.as_list())

    if tf.rank(x).numpy() == 4:
        logger.debug("Kernel: %s", kernel_dims)
        convolutional_layer = tf.keras.layers.Conv2D(
            filters=n_filter,
            kernel_size=kernel_dims,
            strides=stride,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            kernel_regularizer=regularizer,
        )
        x = convolutional_layer(x)
    elif tf.rank(x).numpy() == 5:
        logger.debug("Kernel: %s", kernel_dims)
        convolutional_layer = tf.keras.layers.Conv3D(
            filters=n_filter,
            kernel_size=kernel_dims,
            strides=stride,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            kernel_regularizer=regularizer,
        )
        x = convolutional_layer(x)

    x = activation(act_func)(x)

    logger.debug("Output: %s", x.shape)

    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)

    if drop_out[0]:
        # ToDo: change between 2D and 3D based on rank of x
        if tf.rank(x).numpy() == 4:
            x = tf.keras.layers.SpatialDropout2D(drop_out[1])(x)
        elif tf.rank(x).numpy() == 5:
            x = tf.keras.layers.SpatialDropout3D(drop_out[1])(x)

    return x


def downscale(
    x: tf.Tensor,
    downscale_method: str,
    kernel_dims: Union[int, Tuple],
    n_filter: int,
    stride: Union[int, Tuple],
    padding: str,
    dilation_rate: Union[int, Tuple],
    act_func: str,
    use_bias: bool,
    regularizer: Optional[tf.keras.regularizers.Regularizer],
    cross_hair,
) -> tf.Tensor:
    """!
    Implements a downscale layer: downscale + activation


    net :
        A Network object.
    x :
        A Tensor of TODO(shape,type) : Input Tensor to the block.
    kernel_dims :
        A list of ints : [filter_height, filter_width] of spatial filters of the layer.
    n_filter :
        int : The number of filters of the block.
    @return A Tensor `x`

    This function does the following:
    - perform downscale on `x`
        - If downscale_method is  'STRIDE' , `x` is passed through a convolution operation
          with stride [2,2] (See operation.convolution() )
        - If downscale_method is  'MAX_POOL' , maxpooling is permformed on `x` using tf.nn.max_pool with
            - value : `x`
            - ksize : [1, 2, 2, 1]
            - strides:  [1, 2, 2, 1]
            - padding : padding=net.options['padding']padding=net.options['padding']

    - passes `x` through an activation operation (See operation.activation() ) and return
    """

    # ToDo: change between 2D and 3D based on rank of x
    if downscale_method == "STRIDE":
        logger.debug("Convolution with Stride")
        logger.debug("Input: %s", x.shape.as_list())
        if tf.rank(x).numpy() == 4:
            logger.debug("Kernel: %s,Stride: %s", kernel_dims, np.multiply(stride, 2))
            convolutional_layer = tf.keras.layers.Conv2D(
                filters=n_filter,
                kernel_size=kernel_dims,
                strides=np.multiply(stride, 2),
                padding=padding,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_regularizer=regularizer,
            )
        elif tf.rank(x).numpy() == 5:
            logger.debug("Kernel: %s,Stride: %s", kernel_dims, np.multiply(stride, 2))
            convolutional_layer = tf.keras.layers.Conv3D(
                filters=n_filter,
                kernel_size=kernel_dims,
                strides=np.multiply(stride, 2),
                padding=padding,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_regularizer=regularizer,
            )

        x = convolutional_layer(x)
        x = activation(act_func)(x)

    elif downscale_method == "MAX_POOL":
        if tf.rank(x).numpy() == 4:
            logger.debug("Max Pool")
            logger.debug("Pool Size: %s", [2, 2])
            logger.debug("Input: %s", x.shape.as_list())

            x = tf.keras.layers.MaxPool2D(
                pool_size=[2, 2], strides=[2, 2], padding=padding
            )(x)
        elif tf.rank(x).numpy() == 5:
            logger.debug("Max Pool")
            logger.debug("Pool Size: %s", [2, 2, 2])
            logger.debug("Input: %s", x.shape.as_list())

            x = tf.keras.layers.MaxPool3D(
                pool_size=[2, 2, 2], strides=[2, 2, 2], padding=padding
            )(x)

    elif downscale_method == "MAX_POOL_ARGMAX":
        logger.debug("Max Pool with Argmax")
        logger.debug("Kernel: %s", [1, 2, 2, 1])
        logger.debug("Input: %s", x.shape.as_list())

    logger.debug("Output: %s", x.shape)

    return x


def upscale(
    x: tf.Tensor,
    upscale_method: str,
    kernel_dims: Union[int, Tuple],
    n_filter: int,
    stride: Union[int, Tuple],
    padding: str,
    dilation_rate: Union[int, Tuple],
    act_func: str,
    use_bias: bool,
    regularizer: Optional[tf.keras.regularizers.Regularizer],
    cross_hair,
) -> tf.Tensor:
    """!
    Implements a upcale layer: upcale + activation

    This function does the following:
    - perform downscale on `x`
        - If upscale_method is  'TRANS_CONV'
            - `x` is passed through a transposed convolution operation with
              stride [1, 2, 2, 1] (See operation.transposed_convolution() )
        - If 'upscale_method' is  'UNPOOL_MAX_IND' ,
            - x is passed through a unpooling at indices operation with
                -output shape, outshape given by unpool_param[0] (storing the input shape) and
                - indices, indices given by unpool_param[1] (storing  the maxpool indices)
            of respective maxpooling layer (See operation.unpool_at_indices).
    - passes `x` through an activation operation (See operation.activation() ) and return

    @todo BI_INTER, unpool_param
    """

    if not isinstance(stride, tuple):
        stride = (stride,) * (tf.rank(x).numpy() - 2)

    if upscale_method == "TRANS_CONV":

        strides = np.multiply(stride, 2)

        logger.debug("Transposed Convolution")
        logger.debug("Kernel: %s Stride: %s", kernel_dims, strides)
        logger.debug("Input: %s", x.shape.as_list())

        if tf.rank(x).numpy() == 4:
            convolutional_layer = tf.keras.layers.Conv2DTranspose(
                filters=n_filter,
                kernel_size=kernel_dims,
                strides=strides,
                padding=padding,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_regularizer=regularizer,
            )
            x = convolutional_layer(x)
        elif tf.rank(x).numpy() == 5:
            convolutional_layer = tf.keras.layers.Conv3DTranspose(
                filters=n_filter,
                kernel_size=kernel_dims,
                strides=strides,
                padding=padding,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_regularizer=regularizer,
            )
            x = convolutional_layer(x)

        x = activation(act_func)(x)

    elif upscale_method == "BI_INTER":
        raise NotImplementedError("BI_INTER not implemented")

    elif upscale_method == "UNPOOL_MAX_IND":
        raise NotImplementedError("UNPOOL_MAX_IND not implemented")

    logger.debug("Output: %s", x.shape)
    return x


def last(
    x: tf.Tensor,
    kernel_dims: Union[int, Tuple],
    n_filter: int,
    stride: Union[int, Tuple],
    padding: str,
    dilation_rate,
    act_func: str,
    use_bias: bool,
    regularizer: Optional[tf.keras.regularizers.Regularizer],
    l2_normalize=False,
) -> tf.Tensor:
    """!
    Implements a last layer computing logits

    This function does the following:
        - passes `x` through a convolution operation with stride [1,1] (See operation.convolution() )

    @todo BI_INTER
    """

    logger.debug("Convolution")
    logger.debug("Input: %s", x.shape.as_list())

    if tf.rank(x).numpy() == 4:
        logger.debug("Kernel: %s", kernel_dims)
        convolutional_layer = tf.keras.layers.Conv2D(
            filters=n_filter,
            kernel_size=kernel_dims,
            strides=stride,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            kernel_regularizer=regularizer,
        )
        x = convolutional_layer(x)

    elif tf.rank(x).numpy() == 5:

        logger.debug("Kernel: %s", kernel_dims)
        convolutional_layer = tf.keras.layers.Conv3D(
            filters=n_filter,
            kernel_size=kernel_dims,
            strides=stride,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            kernel_regularizer=regularizer,
        )
        x = convolutional_layer(x)

    x = activation(act_func)(x)

    if l2_normalize:
        x = tf.math.l2_normalize(x, axis=-1)

    return x
