"""
Implementation of DenseNets, right now only the DenseTiramisu
"""
import logging

import tensorflow as tf
from tensorflow.keras import Model

from .utils import get_regularizer, select_final_activation

# configure logger
logger = logging.getLogger(__name__)


def conv_layer(
    x, filters: int, size=3, activation="relu", regularizer=None, drop_out=None, name="conv"
):
    """
    Forms the atomic layer of the tiramisu, does three operation in sequence:
    batch normalization -> Relu -> 2D/3D Convolution.

    Parameters
    ----------
    x: Tensor
        input feature map.
    filters: int
        indicating the number of filters in the output feat. map.
    size : int, optional
        The kernel size (the same will be used in all dimensions), by default 3
    activation : str, optional
        which activation should be used, by default 'relu'
    regularizer : tf.keras.regularizers.Regularizer, optional
        the regularizer to use (or None), by default None
    drop_out : tuple, optional
        if dropout should be used, by default (True, 0.2)
    name: str, optional
        name of the layer, by default conv

    Returns
    -------
    Tensor
        Result of applying batch norm -> Relu -> Convolution.
    """
    rank = len(x.shape) - 2

    if rank == 2:
        conv_layer_type = tf.keras.layers.Conv2D
        dropout_layer_type = tf.keras.layers.SpatialDropout2D
    elif rank == 3:
        conv_layer_type = tf.keras.layers.Conv3D
        dropout_layer_type = tf.keras.layers.SpatialDropout3D
    else:
        raise NotImplementedError("Rank should be 2 or 3")

    bn_layer = tf.keras.layers.BatchNormalization(name=name + "/bn")
    x = bn_layer(x)

    activation_layer = tf.keras.layers.Activation(activation, name=name + "/act")
    x = activation_layer(x)

    convolutional_layer = conv_layer_type(
        filters=filters,
        kernel_size=size,
        strides=(1,) * rank,
        padding="same",
        dilation_rate=(1,) * rank,
        activation=None,
        use_bias=False,
        kernel_regularizer=regularizer,
        name=name + f"/conv{rank}d",
    )
    x = convolutional_layer(x)

    if drop_out[0]:
        dropout_layer = dropout_layer_type(rate=drop_out[1], name=name + "/dropout")
        x = dropout_layer(x)

    return x


def dense_block(
    x,
    n_layers: int,
    growth_rate=16,
    size=3,
    activation="relu",
    regularizer=None,
    drop_out=None,
    name="dense_block",
):
    """
    Forms the dense block of the Tiramisu to calculate features at a specified growth rate.
    Each conv layer in the dense block calculate self.options['growth_rate'] feature maps,
    which are sequentially concatenated to build a larger final output.

    Parameters
    ----------
    x: Tensor
        input to the Dense Block.
    n_layers: int
        the number of layers in the block
    growth_rate : int, optional
        the growth rate in the dense blocks, by default 16
    size : int, optional
        The kernel size (the same will be used in all dimensions), by default 3
    activation : str, optional
        which activation should be used, by default 'relu'
    regularizer : tf.keras.regularizers.Regularizer, optional
        the regularizer to use (or None), by default None
    drop_out : tuple, optional
        if dropout should be used, by default (True, 0.2)
    name: str, optional
        name of the layer, by default dense_block

    Returns
    -------
    Tensor
        the output of the dense block.
    """
    rank = len(x.shape) - 2

    layer_outputs = []
    for i in range(n_layers):
        conv = conv_layer(
            x,
            growth_rate,
            name=name + f"/conv{i}",
            size=size,
            regularizer=regularizer,
            drop_out=drop_out,
            activation=activation,
        )
        layer_outputs.append(conv)
        if i != n_layers - 1:
            concat_layer = tf.keras.layers.Concatenate(
                axis=rank + 1, name=name + f"/concat{i}"
            )
            x = concat_layer([conv, x])

    final_concat_layer = tf.keras.layers.Concatenate(
        axis=rank + 1, name=name + "/concat_conv"
    )
    x = final_concat_layer(layer_outputs)
    return x


def transition_down(
    x, filters: int, activation="relu", regularizer=None, drop_out=None, name="down"
):
    """
    Down-samples the input feature map by half using maxpooling.

    Parameters
    ----------
    x: Tensor
        input to downsample.
    filters: int
        number of output filters.
    activation : str, optional
        which activation should be used, by default 'relu'
    regularizer : tf.keras.regularizers.Regularizer, optional
        the regularizer to use (or None), by default None
    drop_out : tuple, optional
        if dropout should be used, by default (True, 0.2)
    name: str, optional
        name of the layer, by default down

    Returns
    -------
    Tensor
        result of downsampling.
    """
    rank = len(x.shape) - 2

    if rank == 2:
        conv_layer_type = tf.keras.layers.Conv2D
        maxpool_layer_type = tf.keras.layers.MaxPool2D
        dropout_layer_type = tf.keras.layers.SpatialDropout2D
    elif rank == 3:
        conv_layer_type = tf.keras.layers.Conv3D
        maxpool_layer_type = tf.keras.layers.MaxPool3D
        dropout_layer_type = tf.keras.layers.SpatialDropout3D
    else:
        raise NotImplementedError("Rank should be 2 or 3")

    bn_layer = tf.keras.layers.BatchNormalization(name=name + "/bn")
    x = bn_layer(x)

    activation_layer = tf.keras.layers.Activation(activation, name=name + "/act")
    x = activation_layer(x)

    convolutional_layer = conv_layer_type(
        filters=filters,
        kernel_size=(1,) * rank,
        strides=(1,) * rank,
        padding="same",
        dilation_rate=(1,) * rank,
        activation=None,
        use_bias=False,
        kernel_regularizer=regularizer,
        name=name + f"/conv{rank}d",
    )
    x = convolutional_layer(x)

    if drop_out[0]:
        dropout_layer = dropout_layer_type(rate=drop_out[1], name=name + "/dropout")
        x = dropout_layer(x)

    pooling_layer = maxpool_layer_type(
        pool_size=(2,) * rank, strides=(2,) * rank, name=name + f"/maxpool{rank}d"
    )
    x = pooling_layer(x)

    return x


def transition_up(x, filters: int, size=3, regularizer=None, name="up"):
    """
    Up-samples the input feature maps using transpose convolutions.

    Parameters
    ----------
    x: Tensor
        input feature map to upsample.
    filters: int
        number of filters in the output.
    size : int, optional
        The kernel size (the same will be used in all dimensions), by default 3
    regularizer : tf.keras.regularizers.Regularizer, optional
        the regularizer to use (or None), by default None
    name: str
        name of the layer

    Returns
    -------
    Tensor
        result of up-sampling.
    """
    rank = len(x.shape) - 2

    if rank == 2:
        conv_transpose_layer_type = tf.keras.layers.Conv2DTranspose
    elif rank == 3:
        conv_transpose_layer_type = tf.keras.layers.Conv3DTranspose
    else:
        raise NotImplementedError("Rank should be 2 or 3")

    conv_transpose_layer = conv_transpose_layer_type(
        filters=filters,
        kernel_size=size,
        strides=(2,) * rank,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name=name + "_trans_up",
    )
    x = conv_transpose_layer(x)

    return x


# Original names are used for better readability pylint: disable=invalid-name
def DenseTiramisu(
    input_tensor: tf.Tensor,
    out_channels: int,
    loss: str,
    is_training=True,
    kernel_dims=3,
    growth_rate=16,
    layers_per_block=(4, 5, 7, 10, 12),
    bottleneck_layers=15,
    drop_out=(True, 0.2),
    regularize=(True, "L2", 0.001),
    do_batch_normalization=True,
    do_bias=False,
    activation="relu",
    model=Model,
) -> Model:
    """Initialize the 100 layers Tiramisu
    see https://arxiv.org/abs/1611.09326

    Parameters
    ----------
    input_tensor : tf.Tensor
        the tensor used as input, used to derive the rank and the output will have
        the same shape except the input channels will be replaced by the output ones.
        the spatial dimensions have to be divisible by 2**len(layers_per_block)
    out_channels : int
        The number of output channels to use (number of classes + background)
    loss : str
        the type of loss to use
    is_training : bool, optional
        if in training, by default True
    kernel_dims : int, optional
        the dimensions of the kernel (dimension is automatic), by default 3
    growth_rate : int, optional
        the growth rate in the dense blocks, by default 16
    layers_per_block : tuple, optional
        number of layers per block, by default (4, 5, 7, 10, 12) (from paper)
    bottleneck_layers : int, optional
        number of layers in the bottleneck, by default 15 (from paper)
    drop_out : tuple, optional
        if dropout should be used, by default (True, 0.2) (from paper)
    regularize : tuple, optional
        if there should be regularization, by default (True, 'L2', 0.001)
    do_batch_normalization : bool, optional
        has to be true for dense nets, by default True
    do_bias : bool, optional
        has to be false because batch norm, by default False
    activation : str, optional
        which activation should be used, by default 'relu'
    model : tf.keras.Model, optional
        Which model should be used for generation, can be used to use a sub-
        classed model with custom functionality, by default tf.keras.Model
    """

    assert is_training, "only use this for training"

    if do_bias:
        print("use no bias with this network, bias set to False")
        do_bias = False
    if not do_batch_normalization:
        print("always uses batch norm, set to True")
        do_batch_normalization = True

    rank = len(input_tensor.shape) - 2

    # TODO: parameters for pooling and dilations
    n_blocks = len(layers_per_block)
    con_ax = rank + 1

    # define a standard config
    regularizer = get_regularizer(*regularize)
    std = {"regularizer": regularizer, "drop_out": drop_out, "activation": activation}

    if rank == 2:
        conv_layer_type = tf.keras.layers.Conv2D
    elif rank == 3:
        conv_layer_type = tf.keras.layers.Conv3D
    else:
        raise NotImplementedError("Rank should be 2 or 3")

    x = input_tensor
    logger.debug("Start model definition")
    logger.debug("Input Shape: %s", x.get_shape())

    concats = []

    # encoder
    first_layer = conv_layer_type(
        filters=48,
        kernel_size=kernel_dims,
        strides=(1,) * rank,
        padding="same",
        dilation_rate=(1,) * rank,
        kernel_regularizer=regularizer,
        name=f"DT{rank}D-encoder/conv{rank}d",
    )
    x = first_layer(x)
    logger.debug("First Convolution Out: %s", x.get_shape())

    for block_nb in range(0, n_blocks):
        dense = dense_block(
            x,
            layers_per_block[block_nb],
            name=f"DT{rank}D-down_block{block_nb}",
            growth_rate=growth_rate,
            size=kernel_dims,
            **std,
        )
        concat_layer = tf.keras.layers.Concatenate(
            axis=con_ax, name=f"DT{rank}D-concat_output_down{block_nb-1}"
        )
        x = concat_layer([x, dense])
        concats.append(x)
        x = transition_down(
            x, x.get_shape()[-1], name=f"DT{rank}D-transition_down{block_nb}", **std
        )
        logger.debug("Downsample Out: %s", x.get_shape())
        logger.debug("m=%i", x.get_shape()[-1])

    x = dense_block(x, bottleneck_layers, name=f"DT{rank}D-bottleneck", **std)
    logger.debug("Bottleneck Block: %s", x.get_shape())

    # decoder
    for i, block_nb in enumerate(range(n_blocks - 1, -1, -1)):
        logger.debug("Block %i", i)
        logger.debug("Block to upsample: %s", x.get_shape())
        x = transition_up(
            x,
            x.get_shape()[-1],
            size=kernel_dims,
            regularizer=regularizer,
            name=f"DT{rank}D-transition_up{i}",
        )
        logger.debug("Upsample out: %s", x.get_shape())
        concat_layer = tf.keras.layers.Concatenate(
            axis=con_ax, name=f"DT{rank}D-concat_input{i}"
        )
        x_con = concat_layer([x, concats[len(concats) - i - 1]])
        logger.debug("Skip connect: %s", concats[len(concats) - i - 1].get_shape())
        logger.debug("Concat out: %s", x_con.get_shape())
        x = dense_block(
            x_con, layers_per_block[block_nb], name=f"DT{rank}D-up_block{i}", **std
        )
        logger.debug("Dense out: %s", x.get_shape())
        logger.debug("m=%i", x.get_shape()[3] + x_con.get_shape()[3])

    # concatenate the last dense block
    concat_layer = tf.keras.layers.Concatenate(axis=con_ax, name=f"DT{rank}D-last_concat")
    x = concat_layer([x, x_con])

    logger.debug("Last layer in: %s", x.get_shape())

    # prediction
    last_layer = conv_layer_type(
        filters=out_channels,
        kernel_size=(1,) * rank,
        padding="same",
        dilation_rate=(1,) * rank,
        kernel_regularizer=regularizer,
        activation=None,
        use_bias=False,
        name=f"DT{rank}D-prediction/conv{rank}d",
    )
    x = last_layer(x)

    last_activation_layer = tf.keras.layers.Activation(
        select_final_activation(loss, out_channels), name=f"DT{rank}D-prediction/act"
    )
    probabilities = last_activation_layer(x)

    logger.debug("Mask Prediction: %s", x.get_shape())
    logger.debug("Finished model definition.")

    return model(inputs=input_tensor, outputs=probabilities)
