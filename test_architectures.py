# pylint: disable=all

import logging
import sys

import numpy as np
import pytest
import tensorflow as tf

from .deeplab import DeepLabv3plus
from .densenets import DenseTiramisu
from .unets import unet

# from .hrnet import HRNet


@pytest.mark.parametrize(
    "backbone",
    [
        "resnet50",
        "resnet101",
        "resnet152",
        "mobilenet_v2",
        "densenet121",
        "densenet169",
        "densenet201",
        "efficientnetB0",
    ],
)
@pytest.mark.parametrize("in_channels", [1, 3])
def test_deeplab(backbone, in_channels):
    input_shape = (256, 256, in_channels)

    hyperparameters = {"backbone": backbone}

    model_creation(DeepLabv3plus, input_shape, hyperparameters)


@pytest.mark.parametrize("in_channels", [1, 3])
def test_dense_tiramisu(in_channels):
    in_channels = 3
    input_shape = (128, 128, in_channels)
    model_creation(DenseTiramisu, input_shape)


@pytest.mark.parametrize("in_channels", [1])
@pytest.mark.parametrize("batch_norm", [True])
@pytest.mark.parametrize("act_func", ["elu"])
@pytest.mark.parametrize("attention", [False])
@pytest.mark.parametrize("encoder_attention", [None])
@pytest.mark.parametrize("res_connect", [True])
@pytest.mark.parametrize("res_connect_type", ["skip_first"])
@pytest.mark.parametrize("skip_connect", [True])
@pytest.mark.parametrize("name", ["MultiresUnet"])
@pytest.mark.parametrize("n_heads", [2])
@pytest.mark.parametrize("l_layers", [1])
def test_unet(
    in_channels,
    batch_norm,
    act_func,
    attention,
    encoder_attention,
    res_connect,
    res_connect_type,
    skip_connect,
    name,
    n_heads,
    l_layers,
):
    in_channels = 1
    input_shape = (96, 96, in_channels)
    hyperparameters = {
        "batch_normalization": batch_norm,
        "activation": act_func,
        "attention": attention,
        "encoder_attention": encoder_attention,
        "res_connect": res_connect,
        "res_connect_type": res_connect_type,
        "skip_connect": skip_connect,
        "name": name,
        "n_heads": n_heads,
        "l_layers": l_layers,
        "ratio": 2,  # only important for attention models
    }
    model_creation(unet, input_shape, hyperparameters)


# @pytest.mark.parametrize("in_channels", [1, 3])
# def test_HRNet(
#     in_channels,
# ):
#     in_channels = 1
#     input_shape = (96, 96, in_channels)
#
#     model_creation(HRNet, input_shape, hyperparameters)


def model_creation(model, input_shape, hyperparameters={}, do_fit=False, do_plot=True):
    # run on CPU
    with tf.device("/device:CPU:0"):
        out_channels = 2
        batch = 8
        model_built: tf.keras.Model = model(
            tf.keras.Input(shape=input_shape, batch_size=batch, dtype=float),
            out_channels,
            "DICE",
            **hyperparameters,
        )
        output_shape = model_built.output.shape.as_list()
        # make sure that the dimensions are right
        assert output_shape[0] == batch
        assert np.all(np.array(output_shape[1:-1]) == input_shape[:-1])
        assert output_shape[-1] == out_channels
        trainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model_built.trainable_weights]
        )
        nonTrainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model_built.non_trainable_weights]
        )
        print(
            "n_heads and l_layers are:",
            hyperparameters["n_heads"],
            hyperparameters["l_layers"],
        )
        print(
            "\n model:",
            hyperparameters["name"],
            "has total weights:",
            trainableParams + nonTrainableParams,
        )
        model_built.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics="acc",
            optimizer=tf.keras.optimizers.Adam(),
        )
        prediction = model_built.predict(np.zeros((1, 96, 96, 1)), batch_size=1)
        print(prediction.shape)
    print(model_built.summary())
    if do_plot:
        tf.keras.utils.plot_model(model_built, to_file=f"graph-{model.__name__}.png")


if __name__ == "__main__":

    # print all debug messages
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s"
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    for mod in [unet]:
        model_creation(mod, input_shape=(96, 96, 1), do_plot=True)
