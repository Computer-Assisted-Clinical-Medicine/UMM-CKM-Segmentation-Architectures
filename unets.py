"""
Implements multiple different kinds of UNets
"""
from functools import partial
from typing import Callable, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate

from . import layers
from .utils import get_regularizer, select_final_activation


def conv_block(
    x: tf.Tensor,
    n_conv: int,
    conv: Callable,
    n_filter: int,
    attention: Optional[Callable],
    res_connect: bool,
    res_connect_type="skip_first",
) -> tf.Tensor:
    """Convolutional block performing attention and residual connections if
    specified

    Parameters
    ----------
    x : tf.Tensor
        The input
    n_conv : int
        How many convolutions should be performed
    conv : Callable
        The function performing the convolutions, should take x and n_filter as arguments
    n_filter : int
        The number of filters
    attention : Callable
        The attention function, should take x and n_filter as arguments
    res_connect : bool
        If residual connections should be used
    res_connect_type : str, optional
        What should be done if there is a depth-missmatch, by default "skip_first"
        The options are:
        - skip_first : the first convolutional layer will not be included in the res-block
        - 1x1conv : a 1x1 convolution is used to increase the number of channels

    Returns
    -------
    tf.Tensor
        The output tensor
    """
    assert n_conv >= 1, "There must be at least 1 convolution."
    if res_connect:
        x_input = x
    x_first_conv = conv(x, n_filter=n_filter)
    x = x_first_conv
    for _ in range(n_conv - 1):
        x = conv(x, n_filter=n_filter)
    if attention is not None:
        x = attention(x, n_filter=n_filter)
    if res_connect:
        if x.shape == x_input.shape:
            # if shapes match, just add
            x = Add()([x, x_input])
        elif res_connect_type == "skip_first":
            # add the output of the first convolutional layer
            x = Add()([x, x_first_conv])
        elif res_connect_type == "1x1conv":
            # apply a 1x1 convolution to the input to get the right amount of filters
            if tf.rank(x_input) - 2 == 2:
                conv_1 = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=1)
            elif tf.rank(x_input) - 2 == 3:
                conv_1 = tf.keras.layers.Conv3D(filters=n_filter, kernel_size=1)
            x_input_conv = conv_1(x_input)
            x = Add()([x, x_input_conv])
    return x


def encoder_block(
    x: tf.Tensor,
    conv: Callable,
    attention: Optional[Callable],
    downscale: Callable,
    n_conv: int,
    n_filter: int,
    res_connect: bool,
    res_connect_type: str,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encoder block with n_conv convolutional layers followed by downsampling.
    The conv block uses attention and residual connections if specified.

    Parameters
    ----------
    x : tf.Tensor
        The input
    conv : Callable
        The function performing the convolutions, should take x and n_filter as arguments
    attention : Callable
        The attention function, should take x and n_filter as arguments
    downscale : Callable
        The downscale function, should take x and n_filter as arguments
    n_conv : int
        How many convolutions should be performed
    n_filter : int
        The number of filters
    res_connect : bool
        If residual connections should be used
    res_connect_type : str, optional
        What should be done if there is a depth-missmatch, by default "skip_first"
        The options are:
        - skip_first : the first convolutional layer will not be included in the res-block
        - 1x1conv : a 1x1 convolution is used to increase the number of channels

    Returns
    -------
    tf.Tensor
        The output tensor
    tf.Tensor
        The tensor before downscaling (before the skip connection)
    """
    x_before_downscale = conv_block(
        x, n_conv, conv, n_filter, attention, res_connect, res_connect_type
    )
    x = downscale(x_before_downscale, n_filter=n_filter)
    return x, x_before_downscale


def decoder_block(
    x: tf.Tensor,
    x_skip: Optional[tf.Tensor],
    conv: Callable,
    upscale: Callable,
    attention: Optional[Callable],
    gate_signal: Optional[Callable],
    n_conv: int,
    n_filter: int,
    res_connect: bool,
    res_connect_type: str,
) -> tf.Tensor:
    """
    Decoder block, does n_conv convolutions (using the conv function) with
    attention if attention is not none. If res_connect, everything after the
    upscale is wrapped into a residual connection. If x_skip is not none, it is
    concatenated (with attention if not none) with x.

    Parameters
    ----------
    x : tf.Tensor
        The input
    x_skip : tf.Tensor
        The input from the skip connection
    conv : Callable
        The function performing the convolutions, should take x and n_filter as arguments
    upscale : Callable
        The downscale function, should take x and n_filter as arguments
    attention : Callable
        The attention function, should take x and n_filter as arguments
    gate_signal : Callable
        The gate_signal function, should take x as argument
    n_conv : int
        How many convolutions should be performed
    n_filter : int
        The number of filters
    res_connect : bool
        If residual connections should be used
    res_connect_type : str, optional
        What should be done if there is a depth-missmatch, by default "skip_first"
        The options are:
        - skip_first : the first convolutional layer will not be included in the res-block
        - 1x1conv : a 1x1 convolution is used to increase the number of channels

    Returns
    -------
    tf.Tensor
        The output tensor
    """
    x_before_upscale = x
    x = upscale(x, n_filter=n_filter)
    if x_skip is not None:
        if attention is not None:
            assert (
                gate_signal is not None
            ), "If using attention, also provide a gate function"
            gate = tf.identity(x_before_upscale)
            gate = gate_signal(gate)
            attn = attention(x_skip, gate=gate)
            x = Concatenate()([x, attn])
        else:
            x = Concatenate()([x, x_skip])
    # No attention in the conv block
    x = conv_block(x, n_conv, conv, n_filter, None, res_connect, res_connect_type)
    return x


def unet(
    input_tensor: tf.Tensor,
    out_channels: int,
    loss: str,
    n_filter=(8, 16, 32, 64, 128),
    n_convolutions=(2, 2, 3, 3, 3),
    attention=False,
    encoder_attention=None,
    kernel_dims=3,
    stride=1,
    batch_normalization=True,
    use_bias=False,
    drop_out=(False, 0.2),
    upscale="TRANS_CONV",
    downscale="MAX_POOL",
    regularize=(True, "L2", 0.001),
    padding="SAME",
    activation="relu",
    name="Unet",
    ratio=2,
    dilation_rate=1,
    cross_hair=False,
    res_connect=True,
    res_connect_type="skip_first",
    skip_connect=True,
    **kwargs,
) -> tf.keras.Model:
    """
    Implements U-Net (https://arxiv.org/abs/1505.04597) as the backbone. The add-on architectures are Attention U-Net
    (https://arxiv.org/abs/1804.03999), CBAMUnet, CBAMAttnUnet, SEUnet and SEAttnUnet. Where Convolutional block
    attention modules (CBAM - https://arxiv.org/abs/1807.06521) and Squeeze & excitation blocks
    (SE - https://arxiv.org/abs/1709.01507) are added to the encoder of U-Net or Attention U-Net to obtain CBAM and SE
    attention U-Nets respectively.

    If a recognized name is provided (besides the standard name), the  attention parameters
    will be set accordingly. Possible names are:
     - AttnUnet
     - SEUnet
     - SEAttnUnet
     - CBAMUnet
     - CBAMAttnUnet

    Parameters
    ----------
    input_tensor : tf.Tensor
        input tensorflow tensor/image.
    out_channels : int
        number of classes that needs to be segmented.
    loss : str
        loss function as a string
    n_filter : tuple, optional
        a list containing number of filters for conv layers (encoder block: 1 to 5, decoder block: 4 to 1) with the
        last entry being used for the bottleneck layer. By default: (8, 16, 32, 64, 128).
    n_convolutions : tuple, optional
        the number of convolutions to use per block. It should have the same shape as n_filter.
        By default: (2, 2, 3, 3, 3)
    attention : bool, optional
        If attention should be used in the decoding path, by default False
    encoder_attention : str, optional
        If attention should be used in the encoding path, by default None
        Possible values are:
         -None for no attention
         -SE for squeeze and excitation
         -CBAM for Convolutional Block Attention Module
    kernel_dims : int
        shape of all the convolution filter, by default: 3.
    stride : int, optional
        stride for all the conv layers, by default: 1.
    batch_normalization : bool, optional
        boolean value, whether to apply batch_norm or not. By default: True.
    use_bias : bool, optional
        boolean value, whether to apply bias or not. If batch_normalization is true then use_bias must be
        false and vice versa By default: False.
    drop_out : tuple, optional
        a list containing a boolean, whether to apply dropout to conv layers or not. The number signifies
        the probability of dropout. By default: (False, 0.2).
    upscale : str, optional
        The strategy to use for upscaling features. By default: 'TRANS_CONV'.
    downscale : str, optional
        The strategy to downscale features. Options: 'MAX_POOL', 'STRIDE'. By default: 'MAX_POOL'.
    regularize : tuple, optional
        The value for l2 regularization. By default: (True, "L2", 0.001).
    padding : str, optional
        The strategy to pad the features. By default: 'SAME'.
    activation : str, optional
        The activation used after each layer. By default: 'relu'.
    name : str, optional
        The network that the user wants to implement. Must be one of the following: 'Unet', 'SEUnet',
        'SEAttnUnet', 'CBAMUnet', 'CBAMAttnUnet', 'AttnUnet'. By default: Unet.
    ratio : int, optional
        The ratio by which features are reduced in SE or CBAM channel attention, by default 2
    dilation_rate : 1, optional
        dilation rate for convolutions. By default: 1.
    cross_hair : bool, optional
        Boolean, whether to use cross hair convolutions or not. By default: False.
    res_connect : bool, optional
        If residual connections should be used. By default: False
    res_connect_type : str, optional
        The addition in residual connections requires the same number of channels in input and output.
        In the encoding path, the input has less channels. The options are:
        - skip_first : the first convolutional layer will not be included in the res-block
        - 1x1conv : a 1x1 convolution is used to increase the number of channels
        By default: "skip_first"
    skip_connect : bool, optional
        If skip connections should be used. By default: True

    Returns
    -------
    tf.keras.Model
        A model specified in the name argument.
    """
    special_models = [
        "SEUnet",
        "SEAttnUnet",
        "CBAMUnet",
        "CBAMAttnUnet",
        "AttnUnet",
    ]
    # see if the parameters should be inferred
    if name in special_models:
        attention = bool(name in ["AttnUnet", "SEAttnUnet", "CBAMAttnUnet"])
        if name in ["SEUnet", "SEAttnUnet"]:
            encoder_attention = "SE"
        elif name in ["CBAMUnet", "CBAMAttnUnet"]:
            encoder_attention = "CBAM"

    # check the rate if SE or CBAM is used
    if encoder_attention is not None:
        if ratio <= 1:
            raise ValueError("For SE or CBAM blocks to work, use ratio higher than 1")

    if len(n_filter) != len(n_convolutions):
        raise ValueError("n_filter should have the same length as n_convolutions.")

    regularizer = get_regularizer(*regularize)

    # set up permanent arguments of the layers
    stride = [stride] * (tf.rank(input_tensor).numpy() - 2)
    conv = partial(
        layers.convolutional,
        kernel_dims=kernel_dims,
        stride=stride,
        batch_normalization=batch_normalization,
        drop_out=drop_out,
        use_bias=use_bias,
        regularizer=regularizer,
        padding=padding,
        act_func=activation,
        dilation_rate=dilation_rate,
        cross_hair=cross_hair,
    )
    downscale = partial(
        layers.downscale,
        downscale_method=downscale,
        kernel_dims=kernel_dims,
        act_func=activation,
        stride=stride,
        use_bias=use_bias,
        regularizer=regularizer,
        padding=padding,
        dilation_rate=dilation_rate,
        cross_hair=cross_hair,
    )  # here stride is multiplied by 2 in func to downscale by 2
    upscale = partial(
        layers.upscale,
        upscale_method=upscale,
        kernel_dims=kernel_dims,
        act_func=activation,
        stride=stride,
        use_bias=use_bias,
        regularizer=regularizer,
        padding=padding,
        dilation_rate=dilation_rate,
        cross_hair=cross_hair,
    )  # stride multiplied by 2 in function
    gate_signal = partial(
        layers.unet_gating_signal, batch_normalization=batch_normalization
    )
    attn_block = partial(
        layers.attn_gating_block, use_bias=use_bias, batch_normalization=batch_normalization
    )
    se_block = partial(layers.se_block, act_func=activation, ratio=ratio)
    cbam_block = partial(layers.cbam_block, ratio=ratio)

    # input layer
    x = input_tensor

    if encoder_attention is None:
        encoder_attention_func: Optional[partial] = None
    elif encoder_attention == "SE":
        encoder_attention_func = se_block
    elif encoder_attention == "CBAM":
        encoder_attention_func = cbam_block
    else:
        raise ValueError(f"Unknown encoder attention {encoder_attention}")
    # collect the output for skip connections
    skip_connections = []
    # do the encoding path
    for filters, n_conv in zip(n_filter[:-1], n_convolutions[:-1]):
        x, x_skip = encoder_block(
            x=x,
            conv=conv,
            attention=encoder_attention_func,
            downscale=downscale,
            n_conv=n_conv,
            n_filter=filters,
            res_connect=res_connect,
            res_connect_type=res_connect_type,
        )
        skip_connections.append(x_skip)

    # bottleneck layer
    x = conv_block(
        x=x,
        conv=conv,
        n_conv=n_convolutions[-1],
        n_filter=n_filter[-1],
        attention=None,
        res_connect=res_connect,
        res_connect_type=res_connect_type,
    )

    if attention:
        gate_signal_decoder: Optional[partial] = gate_signal
        decode_attention: Optional[partial] = attn_block
    else:
        gate_signal_decoder = None
        decode_attention = None
    # decoding path going the other way in the lists
    for filters, n_conv, x_skip in zip(
        n_filter[-2::-1], n_convolutions[-2::-1], skip_connections[::-1]
    ):
        if not skip_connect:
            x_skip = None
        if decode_attention is None:
            decode_attention_inter_shape = None
        else:
            decode_attention_inter_shape = partial(decode_attention, inter_shape=filters)
        x = decoder_block(
            x=x,
            x_skip=x_skip,
            conv=conv,
            upscale=upscale,
            attention=decode_attention_inter_shape,
            gate_signal=gate_signal_decoder,
            n_conv=n_conv,
            n_filter=filters,
            res_connect=res_connect,
            res_connect_type=res_connect_type,
        )

    # final output layer
    logits = layers.last(
        x,
        kernel_dims=1,
        n_filter=out_channels,
        stride=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        act_func=select_final_activation(loss, out_channels),
        use_bias=False,
        regularizer=regularizer,
        l2_normalize=False,
    )

    return tf.keras.Model(inputs=input_tensor, outputs=logits)
