"""Implements modules for UCTransNet from the paper: https://arxiv.org/pdf/2109.04335.pdf"""
import numpy as np
from typing import Callable, Optional, Tuple, List

import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate, Conv2D, Flatten, Dropout, Reshape, \
    Embedding, Dense, Softmax, LayerNormalization, UpSampling2D, BatchNormalization, ReLU, GlobalAveragePooling2D
import tensorflow_addons as tfa


def channel_embedding(img: tf.Tensor, patch_size: int, img_size: int, in_channels: int) -> tf.Tensor:
    """Calculates channel embedding for a given feature map. The feature map has to be a square matrix in
     height and width

    Parameters
    ----------
    img : tf.Tensor
        Input skip connection feature.
    patch_size : int
        The size to be used for kernel_size and strides to form a feature map into same number of patches for different
        sized feature maps.
    img_size : int
        Height or width dimension of the feature map.
    in_channels : int
        The number of channels in the input feature map

    Returns
    -------
    tf.Tensor
        The output embedding tensor
    """

    patch_size = np.repeat(patch_size, 2)
    img_size = np.repeat(img_size, 2)
    n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

    patch_embedding = Conv2D(filters=in_channels, kernel_size=patch_size, strides=patch_size)
    positional_embedding = Embedding(input_dim=n_patches, output_dim=in_channels)(tf.zeros([1, n_patches], dtype=tf.float32))

    x = patch_embedding(img)
    size_1, size_2, _, size_4 = x.get_shape()
    x = Reshape([size_2 * size_2, size_4])(x)
    # x = tf.transpose(x)

    embedding = x + positional_embedding
    embedding = Dropout(rate=0.2)(embedding)

    return embedding


def create_query(embedding: tf.Tensor) -> tf.Tensor:
    """Create query from tokens/embedding.

    Parameters
    ----------
    embedding : tf.Tensor
        Input embedding for creating query.

    Returns
    ----------
    tf.Tensor
        The output query tensor.
    """

    channel_num = embedding.get_shape()[-1]
    query = Dense(channel_num, activation=None)(embedding)

    return query


def create_key_value(embeddings: List[tf.Tensor]) -> List[tf.Tensor]:
    """"Create a key and value pair from a list of all the embeddings

    Parameters
    ----------
    embeddings: List[tf:Tensor]
        All the embeddings for all the features are used to create key and value pair

    Returns
    -------
    List[tf.Tensor]
        A key and value tensor
    """

    key_value = LayerNormalization()(Concatenate(axis=-1)([embeddings[0], embeddings[1], embeddings[2], embeddings[3]]))
    channel_num = key_value.get_shape()[-1]

    key = Dense(channel_num, activation=None)(key_value)
    value = Dense(channel_num, activation=None)(key_value)

    return key, value


def cross_attention_module(queries: List[tf.Tensor], key: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
    """Calculates cross attention using queries, key and value

    Parameters
    ----------
    queries : List[tf.Tensor]
        List of all the queries.
    key : tf.Tensor
        The key tensor.
    value : tf.Tensor
        The value tensor.

    Returns
    ----------
    List[tf.Tensor]
        A list of cross attention tensors
    """

    queries_transposed = [tf.transpose(query, perm=[0, 2, 1]) for query in queries]
    value_transposed = tf.transpose(value, perm=[0, 2, 1])
    total_channel_num = key.get_shape()[-1]
    cross_attention = []

    for q_transposed in queries_transposed:
        channel_num = q_transposed.get_shape()[-2]  # -2 as channels are shifted in q_transposed
        q_k_prod = tf.math.divide(tf.matmul(q_transposed, key), np.sqrt(total_channel_num))
        q_k_prod = tfa.layers.InstanceNormalization(axis=-1)(q_k_prod)
        q_k_prod = Softmax()(q_k_prod)
        q_k_prod = Dropout(rate=0.2)(q_k_prod)
        output = tf.matmul(q_k_prod, value_transposed)
        output = tf.transpose(output, perm=[0, 2, 1])
        output = tf.math.reduce_mean(output, axis=-1, keepdims=True)
        output = Dense(channel_num, activation=None)(output)
        output = Dropout(rate=0.2)(output)
        cross_attention.append(output)

    return cross_attention


def mlp(input_tensor: tf.Tensor, channel_num: int, mlp_channel: int) -> tf.Tensor:
    """Applies an MLP for creating final output of multi_head attention module

    Parameters
    ----------
    input_tensor : tf.Tensor
        The input tensor.
    channel_num : int
        The number of channels in the input tensor and used as number of filters in last mlp layer.
    mlp_channel : int
        The number of channels to be used as units for first mlp layer.

    Returns
    ----------
    tf.Tensor
        The output tensor.
    """

    kernel_init, bias_init = tf.keras.initializers.GlorotUniform(), tf.random_normal_initializer(stddev=1.0)
    x = Dense(units=mlp_channel, kernel_initializer=kernel_init, bias_initializer=bias_init)(input_tensor)
    x = tfa.activations.gelu(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(units=channel_num, kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = Dropout(rate=0.1)(x)

    return x


def n_layer_transformer(input_features: List[tf.Tensor], patch_sizes: List[int], n_heads: int, l_layers: int = 4,
                        mlp_units: List[int] = [50, 100, 150, 200]) \
        -> List[tf.Tensor]:
    """Performs multi-head attention l_layer number of times to form an l_layer transformer

    Parameters
    ----------
    input_features : List[tf.Tensor]
        The list of input feature maps (skip connections from Unet encoder).
    patch_sizes : List[int]
        A list of integers to be used as patch sizes in channel_embedding() to form embeddings with same 2nd axis dims.
    n_heads : int
        Number of heads in each transformer layer (multi head attention).
    l_layers : int
        Number of transformer layers.
    mlp_units : int
        Number of dense units used in the mlp() function.

    Returns
    ----------
    List[tf.Tensor]
        The output list of output tensors after l_layers of transformer.
    """

    img_sizes = [input_features[0].get_shape()[1], input_features[1].get_shape()[1],
                 input_features[2].get_shape()[1], input_features[3].get_shape()[1]]
    channel_nums = [input_features[0].get_shape()[-1], input_features[1].get_shape()[-1],
                    input_features[2].get_shape()[-1], input_features[3].get_shape()[-1]]

    # create embeddings/tokens for tran
    all_embeddings = [channel_embedding(input_features[i], patch_sizes[i], img_sizes[i], channel_nums[i])
                      for i in range(len(input_features))]

    def multi_head_attention(all_embeddings: List[tf.Tensor], n_heads: int, mlp_units: List[int]) -> List[tf.Tensor]:
        """Creates a multi head attention output for 'n_heads' number of heads

        Parameters
        ----------
        all_embeddings : List[tf.Tensor]
            A list of all the tensor embeddings for creating queries, key and value.
        n_heads : int
            The number heads in a single multi head attention transformer.
        mlp_units : List[int]
            A list of numbers to be used as number of filters in the mlp() function.

        Returns
        ----------
        List[tf.Tensor]
            The list of output tensors
        """

        cross_attns = []
        mult_cross_attn = []

        # create query, key and values and then calculate cross attention for each head
        for i in range(n_heads):
            queries = [create_query(LayerNormalization()(all_embeddings[j])) for j in range(len(all_embeddings))]
            # embeddings are layer normalized inside create_key_value function before producing key and value
            key, value = create_key_value(all_embeddings)
            attentions = cross_attention_module(queries, key, value)
            cross_attns.append(attentions)

        # add all the corresponding cross attentions and divide by n heads
        for i in range(len(all_embeddings)):
            sums = tf.zeros(shape=cross_attns[0][i].get_shape(), dtype=tf.float32)
            for j in range(len(cross_attns)):
                sums += cross_attns[j][i]
            sums /= n_heads
            mult_cross_attn.append(sums)

        # create sum of multi cross attentions and embeddings
        # and produce final output for one layer multi head attention
        sum_embed_mult_attn = [embed + mult_attn for embed, mult_attn in zip(all_embeddings, mult_cross_attn)]
        normed_sum = [LayerNormalization()(sum_i) for sum_i in sum_embed_mult_attn]
        # apply MLP
        normed_sum_mlp = [mlp(normed_sum_i, normed_sum_i.get_shape()[-1], mlp_unit)
                          for normed_sum_i, mlp_unit in zip(normed_sum, mlp_units)]

        outputs = [mlp_sum + embed_attn_sum for mlp_sum, embed_attn_sum in zip(normed_sum_mlp, sum_embed_mult_attn)]

        return outputs

    mult_head_outputs = all_embeddings
    for _ in range(l_layers):
        mult_head_outputs = multi_head_attention(all_embeddings=mult_head_outputs, n_heads=n_heads, mlp_units=mlp_units)

    return mult_head_outputs


# --------------------------------------------------------------------------------
# ----------------------------- decoder part -------------------------------------
def reconstruct(skip_connections: List[tf.Tensor], input_attns: List[tf.Tensor],
                patch_sizes: List[int] = [8, 4, 2, 1], kernel_size: int = 1)\
        -> List[tf.Tensor]:
    """Upsample and do convolution on attention (from transformer) tensors for fusing later with decoder features.

    Parameters
    ----------
    skip_connections : List[tf.Tensor]
        The list of original skip connection features from the unet encoder.
    input_attns : List[tf.Tensor]
        The list of outputs from the transformer attention.
    patch_sizes : List[int]
        The list of sizes that will be used in upsampling layer to recreate tensors same size as skip connections.
    kernel_size : int
        By default is 1. Used for 1x1 convolution.
    Returns
    -------
    List[tf.Tensor]
        The list of reconstructed tensors ready to be fused with decoder features
    """

    batch_size, n_patchs, _ = input_attns[0].get_shape()
    channel_nums = [input_attns[i].get_shape()[-1] for i in range(len(input_attns))]
    height, width = int(np.sqrt(n_patchs)), int(np.sqrt(n_patchs))
    outputs = []
    # reshape the attention outputs
    for i in range(len(input_attns)):
        reshaped_attns = Reshape([height, width, channel_nums[i]])(input_attns[i])
        # upsample the attention features
        reconstruct_attns = UpSampling2D(size=[patch_sizes[i], patch_sizes[i]],
                                         interpolation='bilinear')(reshaped_attns)
        x = Conv2D(filters=channel_nums[i], kernel_size=kernel_size)(reconstruct_attns)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # add input skip connections and x
        out = x + skip_connections[i]
        outputs.append(out)

    return outputs


def cca_decoder_fusion(decoder_feature: tf.Tensor, transformer_output: tf.Tensor ) -> tf.Tensor:
    """Performs channel wise attention and fuse decoder features with transformer outputs.

    Parameters
    ----------
    decoder_feature : tf.Tesnor
        The decoder feature map.
    transformer_output : tf.Tensor
        The reconstructed transformer outputs, having same size as decoder features
    Returns
    ---------
    tf.Tensor
        The output tensor
    """

    channel_num = transformer_output.get_shape()[-1]
    gap_layer = GlobalAveragePooling2D()
    gap_decoder = gap_layer(decoder_feature)
    gap_transformer = gap_layer(transformer_output)

    # mlp layer
    gap_decoder = Dense(channel_num)(gap_decoder)
    gap_transformer = Dense(channel_num)(gap_transformer)
    mask = (gap_decoder + gap_transformer) / 2.0
    mask = tf.math.sigmoid(mask)
    mask = mask[:, tf.newaxis, tf.newaxis, :]

    out = mask * transformer_output
    out = ReLU()(out)
    # concatenate attention features with decoder features
    concat_decoder_transformer = Concatenate(axis=-1)([decoder_feature, out])
    # optionally do 2 layers of conv2d, batchnorm and relu activation
    return concat_decoder_transformer

