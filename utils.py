"""
Different functions to help with the creation of segmentation architectures.
"""
import tensorflow as tf


def get_regularizer(
    regularize: bool, regularizer: str, *params
) -> tf.keras.regularizers.Regularizer:
    """Get the regularizer

    Parameters
    ----------
    regularize : bool
        If a regularizer should be used or not
    regularizer : str
        The type of the regularizer, right now, L1 and L2 are supported
        For those types, the only parameter is the rate.

    Returns
    -------
    tf.keras.regularizers.Regularizer
        The regularizer (or None if regularize is false)
    """
    if regularize:
        if regularizer == "L1":
            return tf.keras.regularizers.l1(params[0])
        elif regularizer == "L2":
            return tf.keras.regularizers.l2(params[0])
        else:
            raise ValueError(regularizer, "is not a supported regularizer .")
    else:
        return None


def select_final_activation(loss: str, out_channels: int) -> str:
    """Select the appropriate final activaten for the loss and number of channels.
    See also http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/

    Parameters
    ----------
    loss : str
        The name of the loss
    out_channels : int
        Number of output channels

    Returns
    -------
    str
        Name of the final activation
    """
    if out_channels > 2 or loss in ["DICE", "TVE", "GDL"]:
        # Dice, GDL and Tversky require SoftMax
        return "softmax"
    elif out_channels == 2 and loss in ["CEL", "WCEL", "GCEL"]:
        return "sigmoid"
    else:
        raise ValueError(
            loss,
            "is not a supported loss function or cannot combined with ",
            out_channels,
            "output channels.",
        )


def diagnose_output(x: tf.Tensor, name="debug") -> tf.Tensor:
    """Diagnose output. This can be added as an intermediate layer which will not
    change the data but will print some diagnostics. It can also be used to set
    breakpoints to access intermediate results during execution

    Parameters
    ----------
    x : tf.Tensor
        The tensor to diagnose
    name : str, optional
        The name, by default "debug"

    Returns
    -------
    tf.Tensor
        The input tensor without changes
    """
    if not tf.executing_eagerly():
        name = x.name
    tf.print(f"Diagnosing {name}")
    tf.print("\t Min: ", tf.reduce_min(x))
    tf.print("\tMean: ", tf.reduce_mean(x))
    tf.print("\t Max: ", tf.reduce_max(x))
    tf.debugging.check_numerics(x, f"Nans in {name}")
    return x


def diagnose_wrapper(x, name="debug"):
    return tf.keras.layers.Lambda(lambda x: diagnose_output(x, name=name), name=name)(x)
