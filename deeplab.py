"""
DeepLab networks, right now only DeepLabv3plus is implemented.
inspired by https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0
and https://github.com/bonlime/keras-deeplab-v3-plus
"""

import logging
from typing import List

import tensorflow as tf
from tensorflow.keras import Model

from .utils import get_regularizer, select_final_activation, diagnose_wrapper

# configure logger
logger = logging.getLogger(__name__)


def configure_backbone(name: str, input_tensor: tf.Tensor):
    """Select the backbone and return it

    Parameters
    ----------
    name : str
        The name of the network (currently, resnet50 is implemented)
    input_tensor : tf.Tensor
        The input tensor

    Returns
    -------
    tf.Model, str, str
        The backbone model and the layer for the low-level and high-level features
        layer_low is reduced by a factor of 4 to the input and layer_high by 16.

    Raises
    ------
    ValueError
        If the input_tensor can not be used by the network
    NotImplementedError
        If a not recognized network is selected
    """
    rank = len(input_tensor.shape) - 2
    if name == "resnet50":
        if rank != 2:
            raise ValueError("ResNet50 Backbone can only be used for 2D networks")
        # should be output after removing the last 1 or 2 blocks (with factor 16 compared to input resolution)
        layer_high = "conv4_block6_out"
        # should be with a factor 4 reduced compared to input resolution
        layer_low = "conv2_block3_out"
        backbone = tf.keras.applications.ResNet50(
            include_top=False, input_tensor=input_tensor
        )
    else:
        raise NotImplementedError(f"Backbone {name} unknown.")

    return backbone, layer_low, layer_high


def upsample(x: tf.Tensor, size: List, name="up") -> tf.Tensor:
    """Do bilinear upsampling of the data

    Parameters
    ----------
    x : tf.Tensor
        The input Tensor
    size : List
        The size which should be upsampled to
    name : str, optional
        Name used for this layer, by default "up"

    Returns
    -------
    tf.Tensor
        The upsampled tensor
    """
    x = tf.image.resize(x, size=size, method=tf.image.ResizeMethod.BILINEAR, name=name)
    return x


def convolution(
    x: tf.Tensor,
    filters: float,
    size=3,
    dilation_rate=None,
    padding="same",
    depthwise_separable=False,
    depth_activation=False,
    activation="relu",
    regularizer=None,
    drop_out=None,
    name="conv",
) -> tf.Tensor:
    """Do a convolution (depthwise if specified). Depthwise convolutions work
    by first applying a convolution to each feature map separately followed
    by a 1x1 convolution.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor
    filters : float
        How many filters should be used
    size : int, optional
        The kernel size (the same will be used in all dimensions), by default 3
    dilation_rate : int, optional
        Rate if using dilated convolutions, by default None
    padding : str, optional
        Which padding to use, by default "same"
    depthwise_separable : bool, optional
        If the convolution should be depthwise separable, by default False
    depth_activation : bool, optional
        If there should be an activation after the depthwise convolution, by default False
    activation : str, optional
        which activation should be used, by default 'relu'
    regularizer : tf.keras.regularizers.Regularizer, optional
        the regularizer to use (or None), by default None
    drop_out : tuple, optional
        if dropout should be used, by default (True, 0.2)
    name : str, optional
        The name of the layer, by default "conv"

    Returns
    -------
    tf.Tensor
        [description]
    """

    if dilation_rate is None:
        dilation_rate = size
    if depthwise_separable:
        # do first depthwise and then pointwise
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=size,
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_regularizer=regularizer,
            use_bias=False,
            name=f"{name}/depthwise-conv",
        )(x)
        if depth_activation:
            x = tf.keras.layers.Activation(activation, name=f"{name}/depthwise-act")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}/depthwise-bn")(x)
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            padding=padding,
            kernel_regularizer=regularizer,
            use_bias=False,
            name=f"{name}/pointwise-conv",
        )(x)
    else:
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=size,
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_regularizer=regularizer,
            use_bias=False,
            name=f"{name}/conv",
        )(x)

    x = tf.keras.layers.Activation(activation, name=f"{name}/act")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}/bn")(x)
    return x


def aspp(
    x: tf.Tensor,
    rates: List,
    filters=256,
    size=3,
    activation="relu",
    regularizer=None,
    drop_out=None,
    name="ASPP",
) -> tf.Tensor:
    """Do atrous convolutions spatial pyramid pooling, the rates are used for
    3x3 convolutions with additional features from a 1x1 convolution and
    global pooling. The number of output features is (2 + len(rates)) * filters.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor
    rates : List
        The atrous (dilation rates)
    filters : int, optional
        How many filters to use per layer, by default 256
    size : int, optional
        The kernel size (the same will be used in all dimensions), by default 3
    activation : str, optional
        which activation should be used, by default 'relu'
    regularizer : tf.keras.regularizers.Regularizer, optional
        the regularizer to use (or None), by default None
    drop_out : tuple, optional
        if dropout should be used, by default (True, 0.2)
    name : str, optional
        The name, by default "ASPP"

    Returns
    -------
    tf.Tensor
        The resulting tensor
    """

    std = {"regularizer": regularizer, "drop_out": drop_out, "activation": activation}

    input_size = tf.keras.backend.int_shape(x)[1:3]
    results = []
    # do 1x1
    results.append(convolution(x, filters, size=1, name=f"{name}/conv_1x1_rate_1"))
    # do 3x3 convs with the corresponding rates
    for r in rates:
        results.append(
            convolution(
                x,
                filters,
                size=size,
                dilation_rate=r,
                depthwise_separable=True,
                depth_activation=True,
                name=f"{name}/conv_3x3_rate_{r}",
                **std,
            )
        )
    # do global average pooling
    pool = tf.keras.layers.GlobalAvgPool2D(name=f"{name}/global_pool/pool")(x)
    # add the dimensions again
    pool = tf.expand_dims(pool, axis=1, name=f"{name}/global_pool/expand0")
    pool = tf.expand_dims(pool, axis=1, name=f"{name}/global_pool/expand1")
    pool = convolution(pool, filters, size=1, name=f"{name}/global_pool/conv", **std)
    results.append(upsample(pool, size=input_size, name=f"{name}/global_pool/up"))
    # concatenate all feature maps
    return tf.keras.layers.Concatenate(name=f"{name}/concat")(results)


# Original names are used for better readability pylint: disable=invalid-name
def DeepLabv3plus(
    input_tensor: tf.Tensor,
    out_channels: int,
    loss: str,
    is_training=True,
    kernel_dims=3,
    drop_out=(True, 0.2),
    regularize=(True, "L2", 0.001),
    backbone="resnet50",
    aspp_rates=(6, 12, 18),
    activation="relu",
    model=Model,
    debug=False,
) -> Model:
    """Build the DeepLabv3plus model

    Parameters
    ----------
    input_tensor : tf.Tensor
        the tensor used as input, used to derive the rank and the output will have
        the same shape except the input channels will be replaced by the output ones.
    out_channels : int
        The number of output channels to use (number of classes + background)
    loss : str
        the type of loss to use
    is_training : bool, optional
        if in training, by default True, does not change anything right now
    kernel_dims : int, optional
        the dimensions of the kernel (dimension is automatic), by default 3
    drop_out : tuple, optional
        if dropout should be used, by default (True, 0.2)
    regularize : tuple, optional
        if there should be regularization, by default (True, 'L2', 0.001)
    activation : str, optional
        which activation should be used, by default 'relu'
    model : tf.keras.Model, optional
        Which model should be used for generation, can be used to use a sub-
        classed model with custom functionality, by default tf.keras.Model
    """

    # TODO: add change in stride to only reduce features by factor 8 (memory intensive), maybe as separate function
    # TODO: add dropout and bias

    regularizer = get_regularizer(*regularize)
    backbone, layer_low, layer_high = configure_backbone(backbone, input_tensor)

    assert is_training, "only use this for training"

    # define a standard config
    std = {
        "regularizer": get_regularizer(*regularize),
        "drop_out": drop_out,
        "activation": activation,
    }

    # make backbone untrainable
    backbone.trainable = False

    x = input_tensor
    input_size = tf.keras.backend.int_shape(x)[1:3]

    logger.debug("Start model definition")
    logger.debug("Input Shape: %s", x.get_shape())

    # for lower features, first reduce number of features with 1x1 conv with 48 filters
    x_low = backbone.get_layer(layer_low).output
    if debug:
        x_low = diagnose_wrapper(x_low, name="x_low_initial_output")
    x_low_size = tf.keras.backend.int_shape(x_low)[1:3]
    x_low = convolution(x_low, filters=48, size=1, name="low-level-reduction", **std)
    if debug:
        x_low = diagnose_wrapper(x_low, name="x_low_after_reduction")

    x_high = backbone.get_layer(layer_high).output
    if debug:
        x_high = diagnose_wrapper(x_high, name="x_high_initial_output")
    x_high = aspp(x_high, aspp_rates, name="ASPP", **std)
    # 1x1 convolution
    x_high = convolution(x_high, size=1, filters=256, name="high-feature-red", **std)
    # upsample to the same size as x_low
    x_high = upsample(x_high, size=x_low_size, name="upsample-high")
    if debug:
        x_high = diagnose_wrapper(x_high, name="x_high_after_upsampling")

    x = tf.keras.layers.Concatenate(name="concat")([x_low, x_high])

    # after concatenation, do two 3x3 convs with 256 filters, BN and act
    x = convolution(
        x,
        filters=256,
        size=kernel_dims,
        depthwise_separable=True,
        depth_activation=True,
        name="pred-conv0",
        **std,
    )
    x = convolution(
        x,
        filters=256,
        size=kernel_dims,
        depthwise_separable=True,
        depth_activation=True,
        name="pred-conv1",
        **std,
    )
    x = upsample(x, size=input_size, name="final-upsample")

    if debug:
        x = diagnose_wrapper(x, name="x_after_final_upsample")

    x = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        padding="same",
        dilation_rate=1,
        kernel_regularizer=regularizer,
        activation=None,
        use_bias=False,
        name="logits",
    )(x)

    probabilities = tf.keras.layers.Activation(
        select_final_activation(loss, out_channels), name="final_activation"
    )(x)

    if debug:
        probabilities = diagnose_wrapper(probabilities, name="probabilities")

    return model(inputs=input_tensor, outputs=probabilities)
