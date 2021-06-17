import tensorflow as tf

def get_regularizer(regularize, regularizer, *params):
    if regularize:
        if regularizer == 'L1':
            regularizer = tf.keras.regularizers.l1(params[0])
        elif regularizer == 'L2':
            regularizer = tf.keras.regularizers.l2(params[0])
        else:
            raise ValueError(regularizer, 'is not a supported regularizer .')
    else:
        regularizer = None

    return regularizer

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
