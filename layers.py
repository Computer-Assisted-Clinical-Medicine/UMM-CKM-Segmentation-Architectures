import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add, multiply, Lambda, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D,\
    Reshape, Dense, Permute, Concatenate, Conv2D, Add
from tensorflow.keras.activations import sigmoid
import numpy as np


VERBOSE = False

def activation(act_func):
    if act_func in tf.keras.activations.__dict__:
       return tf.keras.layers.Activation(act_func)
    elif act_func == 'leaky_relu':
        return tf.keras.layers.LeakyReLU()
    elif act_func == 'swish':
        return swish
    elif act_func == 'elu':
        return tf.keras.layers.ELU()

def swish(x, beta=1):
    return tf.keras.layers.Multiply(x, tf.keras.activations.sigmoid(beta * x))


def expend_as(tensor, rep, axis, name=None):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=axis), arguments={'repnum': rep})(tensor)
                       # name='psi_up' + name)(tensor)

    return my_repeat


def attn_gating_block(x, g, inter_shape, use_bias, batch_normalization, name=None):
    """ take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same feature channels (theta_x)
    then, upsample g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients """

    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    if tf.rank(x).numpy() == 4:
        theta_x = tf.keras.layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', use_bias=use_bias)(x)
                                     # name='xl' + name)(x)  # 16
        shape_theta_x = K.int_shape(theta_x)
        phi_g = tf.keras.layers.Conv2D(inter_shape, (1, 1), padding='same')(g)
        upsample_g = tf.keras.layers.Conv2DTranspose(inter_shape, (3, 3),
                                                     strides=(
                                                         shape_theta_x[1] // shape_g[1],
                                                         shape_theta_x[2] // shape_g[2]),
                                                     padding='same', use_bias=use_bias)(phi_g)  # , name='g_up' + name)(phi_g)  # 16
        # upsample_g = tf.keras.layers.UpSampling2D((shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
        #                                              interpolation='bilinear')(phi_g)  # , name='g_up' + name)(phi_g)  # 16
        concat_xg = add([upsample_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = tf.keras.layers.Conv2D(1, (1, 1), padding='same', use_bias=use_bias)(
            act_xg)  # , name='psi' + name)(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = tf.keras.layers.UpSampling2D(
            size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
        upsample_psi = expend_as(upsample_psi, shape_x[3], axis=3)
        y = multiply([upsample_psi, x])  # , name='q_attn' + name)
        result = tf.keras.layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)  # , name='q_attn_conv' + name)(y)

    elif tf.rank(x).numpy() == 5:
        theta_x = tf.keras.layers.Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), padding='same', use_bias=use_bias)(x)
        # name='xl' + name)(x)  # 16
        shape_theta_x = K.int_shape(theta_x)
        phi_g = tf.keras.layers.Conv3D(inter_shape, (1, 1, 1), padding='same')(g)
        upsample_g = tf.keras.layers.Conv3DTranspose(inter_shape, (3, 3, 3),
                                                     strides=(
                                                         shape_theta_x[1] // shape_g[1],
                                                         shape_theta_x[2] // shape_g[2],
                                                         shape_theta_x[3] // shape_g[3]),
                                                     padding='same', use_bias=use_bias)(phi_g)  # , name='g_up' + name)(phi_g)  # 16
        # upsample_g = tf.keras.layers.UpSampling3D((shape_theta_x[1] // shape_g[1],
        #                                            shape_theta_x[2] // shape_g[2],
        #                                            shape_theta_x[3] // shape_g[3]))(phi_g)  # , name='g_up' + name)(phi_g)  # 16
        concat_xg = add([upsample_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = tf.keras.layers.Conv3D(1, (1, 1, 1), padding='same', use_bias=use_bias)(
            act_xg)  # , name='psi' + name)(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = tf.keras.layers.UpSampling3D(
            size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))\
            (sigmoid_xg)  # 32
        upsample_psi = expend_as(upsample_psi, shape_x[4], axis=4)
        y = multiply([upsample_psi, x])  # , name='q_attn' + name)
        result = tf.keras.layers.Conv3D(shape_x[4], (1, 1, 1), padding='same')(y)  # , name='q_attn_conv' + name)(y)

    if batch_normalization:
        result = tf.keras.layers.BatchNormalization()(result)  # name='q_attn_bn' + name)(result)

    return result


def unet_gating_signal(input, batch_normalization, name=None):
    """ this is simply 1x1 convolution, bn, activation used for calculating gating signal for attention block. """

    shape = K.int_shape(input)
    if tf.rank(input).numpy() == 4:
        x = tf.keras.layers.Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same"# , kernel_initializer=kinit
                                   # , name=name + '_conv'
                                   )(input)
    elif tf.rank(input).numpy() == 5:
        x = tf.keras.layers.Conv3D(shape[4] * 1, (1, 1, 1), strides=(1, 1, 1), padding="same"# , kernel_initializer=kinit
                                   # , name=name + '_conv'
                                   )(input)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)  # name=name + '_bn')(x)
    x = Activation('relu')(x)  # , name=name + '_act')(x)

    return x


def se_block(in_block, channels, activation, ratio):
    """Adds a channelwise attention to the layer called squeeze and excitation block.

    :param in_block: input feature map
    :param channels: number of desire output channels, preferrably same as the channels in in_block.
    :param ratio: ratio to divide the channel number for excitation.

    :return: scaled input with the se matrix.
    """

    if tf.rank(in_block).numpy() == 4:
        x = tf.keras.layers.GlobalAveragePooling2D()(in_block)
        x = tf.keras.layers.Dense(channels // ratio, activation=activation)(x)
        x = tf.keras.layers.Dense(channels, activation='sigmoid')(x)
    else:
        # x = tf.keras.layers.GlobalAveragePooling3D()(in_block)
        num_slices = in_block.shape[1]
        x = tfa.layers.AdaptiveAveragePooling3D((num_slices, 1, 1))(in_block)
        x = tf.keras.layers.Dense(channels // ratio, activation=activation)(x)
        x = tf.keras.layers.Dense(channels, activation='sigmoid')(x)

    return multiply([in_block, x])


def cbam_block(cbam_feature, ratio=4):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    """part of cbam attention module"""

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='elu',  # try elu afterwards
                             #kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             #kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros')

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
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    """part of cbam attention module"""

    kernel_size = 5

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          #kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def convolutional(x, filter_shape, n_filter, stride, padding, dilation_rate,
                  act_func, use_bias, batch_normalization, drop_out, regularizer, cross_hair, do_summary=False):
    '''!
    Implements a convolutional layer: convolution + activation

    Given an input tensor `x` of shape **TODO**, a filter kernel shape `filter_shape`,
    the number of filters `n_filter`, this function performs the following,
        - passes `x` through a convolution operation with stride \[1,1\] (See operation.convolution() )
        - passes `x` through an activation operation (See operation.activation() ) and return


    @param net              A Network object.
    @param x                A Tensor of TODO(shape,type) : Input Tensor to the block.
    @param filter_shape     A list of ints : [filter_height, filter_width] of spatial filters of the layer.
    @param n_filter         int : The number of filters of the block.
    @param drop_out         if True, do dropout
    @param do_summary       TODO
    @return A Tensor `x`

    @todo do_summary
    '''

    if VERBOSE:
        print('   Convolution')
        print('    Input: ', x.shape.as_list())

    if tf.rank(x).numpy() == 4:
        print('    Kernel: ', filter_shape)
        convolutional_layer = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=filter_shape, strides=stride,
                                                     padding=padding,
                                                     dilation_rate=dilation_rate, use_bias=use_bias,
                                                     kernel_regularizer=regularizer)
        x = convolutional_layer(x)
    elif tf.rank(x).numpy() == 5:
            print('    Kernel: ', filter_shape)
            convolutional_layer = tf.keras.layers.Conv3D(filters=n_filter, kernel_size=filter_shape, strides=stride,
                                                         padding=padding,
                                                         dilation_rate=dilation_rate,
                                                         use_bias=use_bias, kernel_regularizer=regularizer)
            x = convolutional_layer(x)

    x = activation(act_func)(x)

    # if do_summary:
    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram('weights', convolutional_layer.get_weights()[0])
    #         tf.summary.histogram('activations', x)
    #         if use_bias:
    #                 with tf.device('/cpu:0'):
    #                     tf.summary.histogram('biases', convolutional_layer.get_weights()[1])

    if VERBOSE:
        print('    Output: ', x.shape)

    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)

    if drop_out[0]:
        # ToDo: change between 2D and 3D based on rank of x
        if tf.rank(x).numpy() == 4:
            x = tf.keras.layers.SpatialDropout2D(drop_out[1])(x)
        elif tf.rank(x).numpy() == 5:
            x = tf.keras.layers.SpatialDropout3D(drop_out[1])(x)

    return x

def downscale(x, downscale, filter_shape, n_filter, stride, padding, dilation_rate,
              act_func, use_bias, regularizer, cross_hair, do_summary=False):
    '''!
    Implements a downscale layer: downscale + activation


    @param net              A Network object.
    @param x                A Tensor of TODO(shape,type) : Input Tensor to the block.
    @param filter_shape     A list of ints : [filter_height, filter_width] of spatial filters of the layer.
    @param n_filter         int : The number of filters of the block.
    @param do_summary       TODO
    @return A Tensor `x`

    This function does the following:
    - perform downscale on `x`
        - If <tt>net.options['downscale']</tt> is <tt> 'STRIDE' </tt>, `x` is passed through a convolution operation with stride \[2,2\] (See operation.convolution() )
        - If <tt>net.options['downscale']</tt> is <tt> 'MAX_POOL' </tt>, maxpooling is permformed on `x` using [tf.nn.max_pool](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool) with
            - @b value : `x`
            - @b ksize : <tt>[1, 2, 2, 1]</tt>
            - @b strides: <tt> [1, 2, 2, 1]</tt>
            - @b padding : <tt>padding=net.options['padding']padding=net.options['padding']</tt>

    - passes `x` through an activation operation (See operation.activation() ) and return

    @todo do_summary
    '''

    # ToDo: change between 2D and 3D based on rank of x
    if downscale == 'STRIDE':
        if VERBOSE:
            print('   Convolution with Stride')
            print('    Input: ', x.shape.as_list())
        if tf.rank(x).numpy() == 4:
            print('    Kernel: ', filter_shape, ',Stride: ', np.multiply(stride, 2))
            convolutional_layer = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=filter_shape,
                                                         strides=np.multiply(stride, 2),
                                                         padding=padding, dilation_rate=dilation_rate,
                                                         use_bias=use_bias, kernel_regularizer=regularizer)
        elif tf.rank(x).numpy() == 5:
            print('    Kernel: ', filter_shape, ',Stride: ', np.multiply(stride, 2))
            convolutional_layer = tf.keras.layers.Conv3D(filters=n_filter, kernel_size=filter_shape,
                                                         strides=np.multiply(stride, 2),
                                                         padding=padding, dilation_rate=dilation_rate,
                                                         use_bias=use_bias, kernel_regularizer=regularizer)

        x = convolutional_layer(x)
        x = activation(act_func)(x)

        # if do_summary:
        #     with tf.device('/cpu:0'):
        #         tf.summary.histogram('weights', convolutional_layer.get_weights()[0])
        #         if use_bias:
        #             with tf.device('/cpu:0'):
        #                 tf.summary.histogram('biases', convolutional_layer.get_weights()[1])

    elif downscale == 'MAX_POOL':
        if tf.rank(x).numpy() == 4:
            if VERBOSE:
                print('   Max Pool')
                print('    Pool Size: ', [2, 2])
                print('    Input: ', x.shape.as_list())

            x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding=padding)(x)
        elif tf.rank(x).numpy() == 5:
            if VERBOSE:
                print('   Max Pool')
                print('    Pool Size: ', [2, 2, 2])
                print('    Input: ', x.shape.as_list())

            x = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding=padding)(x)

    elif downscale == 'MAX_POOL_ARGMAX':
        if VERBOSE:
            print('   Max Pool with Argmax')
            print('    Kernel: ', [1, 2, 2, 1])
            print('    Input: ', x.shape.as_list())
        # # We need to store input shape of this maxpooling layer to use as output shape of corresponding the unpooling upscale layer
        # input_shape = x.shape.as_list()
        # x, max_indices = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                                             padding=options['padding'])
        #
        # # store both input_shape and max_indices to use as params for unpooling at max_indices upscale layer.
        # unpool_params = [input_shape, max_indices]
        #     x = op.convolution(x, filter_shape, n_filter, [2, 2], options['padding'], options['dilation_rate'],
        #                        do_summary)

    # if do_summary:
    #     with tf.device('/cpu:0'):
    #         tf.summary.histogram('activations', x)

    if VERBOSE:
        print('    Output: ', x.shape)

    return x

def upscale(x, upscale, filter_shape, n_filter, stride, padding, dilation_rate,
            act_func, use_bias, regularizer, cross_hair, do_summary=False):
    '''!
    Implements a upcale layer: upcale + activation


    @param net              A Network object.
    @param x                A Tensor of TODO(shape,type) : Input Tensor to the block.
    @param filter_shape     A list of ints : [filter_height, filter_width] of spatial filters of the layer.
    @param n_filter         int : The number of filters of the block.
    @param unpool_param      todo

    @return A Tensor `x`

    This function does the following:
    - perform downscale on `x`
        - If <tt>net.options['upscale']</tt> is <tt> 'TRANS_CONV' </tt>, `x` is passed through a transposed convolution operation with stride <tt> [1, 2, 2, 1]</tt> (See operation.transposed_convolution() )
        - If <tt>net.options['upscale']</tt> is <tt> 'BI_INTER' </tt>, @b TODO
        - If <tt>net.options['upscale']</tt> is <tt> 'UNPOOL_MAX_IND' </tt>,
            - @em x is passed through a unpooling at indices operation with
                -output shape, @em outshape given by <em> unpool_param\[0\]</em> (storing the input shape) and
                - indices, @em indices given by <em> unpool_param\[1\]</em> (storing  the maxpool indices)
            of respective maxpooling layer (See operation.unpool_at_indices).
            - @em x is passed through a convolution operation with @b TODO
    - passes `x` through an activation operation (See operation.activation() ) and return

    @todo do_summary, BI_INTER, unpool_param
    '''

    if upscale == 'TRANS_CONV':

        strides = np.multiply(stride, 2)

        if VERBOSE:
            print('   Transposed Convolution')
            print('    Kernel: ', filter_shape, ' Stride: ', strides)
            print('    Input: ', x.shape.as_list())

        if tf.rank(x).numpy() == 4:
            convolutional_layer = tf.keras.layers.Conv2DTranspose(filters=n_filter, kernel_size=filter_shape,
                                                                  strides=strides, padding=padding,
                                                                  dilation_rate=dilation_rate,
                                                                  use_bias=use_bias, kernel_regularizer=regularizer)
            x = convolutional_layer(x)
        elif tf.rank(x).numpy() == 5:

            convolutional_layer = tf.keras.layers.Conv3DTranspose(filters=n_filter, kernel_size=filter_shape,
                                                                  strides=strides, padding=padding,
                                                                  dilation_rate=dilation_rate,
                                                                  use_bias=use_bias, kernel_regularizer=regularizer)
            x = convolutional_layer(x)

        x = activation(act_func)(x)

        # if do_summary:
        #     with tf.device('/cpu:0'):
        #         tf.summary.histogram('weights', convolutional_layer.get_weights()[0])
        #         if use_bias:
        #             with tf.device('/cpu:0'):
        #                 tf.summary.histogram('biases', convolutional_layer.get_weights()[1])

    elif upscale == 'BI_INTER':
        pass
        # s = x.shape.as_list()
        # # resize each channel
        # new_size = np.multiply(s[1:3], 2)
        # x = tf.image.resize_images(x, new_size)
        # if VERBOSE:
        #     print('  ', 'Upscale (Bilinear Interpolation) to ', x.shape)
        # # apply convolution to reduce number of channels
        # x = op.convolution(x, filter_shape, n_filter, [1, 1], options['padding'], options['dilation_rate'], do_summary)

    elif upscale == 'UNPOOL_MAX_IND':
        # # s = x.shape.as_list()
        # # apply convolution to reduce number of channels
        # x = op.convolution(x, filter_shape, n_filter, [1, 1], options['padding'], options['dilation_rate'], do_summary)
        # x = op.unpool_at_indices(x, outshape=variables['unpool_param'][0], indices=variables['unpool_param'][1])
        if VERBOSE:
            print('  ', 'Upscale (Unpool with Maxpool Indices) to ', x.shape)

    if VERBOSE:
        print('    Output: ', x.shape)
    return x


def last(x, outputs, filter_shape, n_filter, stride, padding, dilation_rate,
         act_func, use_bias, regularizer, cross_hair, do_summary=False, l2_normalize=False):
    '''!
    Implements a last layer computing logits


    @param net              A Network object.
    @param x                A Tensor of TODO(shape,type) : Input Tensor to the block.
    @param filter_shape     A list of ints : [filter_height, filter_width] of spatial filters of the layer.
    @param n_filter         int : The number of filters of the block.
    @return A Tensor `x`

    This function does the following:
        - passes `x` through a convolution operation with stride \[1,1\] (See operation.convolution() )
        - If <tt>net.options['use_bias']</tt> is <tt> True </tt>, @b todo, (See [ tf.nn.bias_add](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add))

    @todo do_summary, BI_INTER
    '''

    if VERBOSE:
        print('   Convolution')
        print('    Input: ', x.shape.as_list())

    if tf.rank(x).numpy() == 4:
        print('    Kernel: ', filter_shape)
        convolutional_layer = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=filter_shape, strides=stride,
                                                     padding=padding, dilation_rate=dilation_rate,
                                                     use_bias=use_bias, kernel_regularizer=regularizer)
        x = convolutional_layer(x)

    elif tf.rank(x).numpy() == 5:

        print('    Kernel: ', filter_shape)
        convolutional_layer = tf.keras.layers.Conv3D(filters=n_filter, kernel_size=filter_shape, strides=stride,
                                                     padding=padding, dilation_rate=dilation_rate,
                                                     use_bias=use_bias, kernel_regularizer=regularizer)
        x = convolutional_layer(x)

    outputs['logits'] = x

    x = activation(act_func)(x)

    if l2_normalize:
        x = tf.math.l2_normalize(x, axis=-1)

    return x


def select_final_activation(loss, out_channels):
    # http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
    if out_channels > 2 or loss in ['DICE', 'TVE', 'GDL']:
        # Dice, GDL and Tversky require SoftMax
        return 'softmax'
    elif out_channels == 2 and loss in ['CEL', 'WCEL', 'GCEL']:
        return 'sigmoid'
    else:
        raise ValueError(loss, 'is not a supported loss function or cannot combined with ',
                            out_channels, 'output channels.')