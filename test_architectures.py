# pylint: disable=all

import logging
import sys

import numpy as np
import pytest
import tensorflow as tf

from .deeplab import DeepLabv3plus
from .densenets import DenseTiramisu
from .unets import unet, unet_old

def test_deeplab():
    in_channels = 3
    input_shape = (256, 256, in_channels)
    model_creation(DeepLabv3plus, in_channels, input_shape)

@pytest.mark.parametrize("in_channels", [1, 2, 3])
def test_dense_tiramisu(in_channels):
    in_channels = 3
    input_shape = (128, 128, in_channels)
    model_creation(DeepLabv3plus, in_channels, input_shape)

@pytest.mark.parametrize("model", [unet, unet_old])
@pytest.mark.parametrize("in_channels", [1, 2, 3])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("act_func", ["relu", "elu"])
@pytest.mark.parametrize("name", ["Unet", "SEUnet", "SEAttnUnet", "CBAMUnet", "CBAMAttnUnet", "AttnUnet"])
@pytest.mark.parametrize("res_connect", [True, False])
@pytest.mark.parametrize("res_connect_type", ["skip_first", "1x1conv"])
@pytest.mark.parametrize("skip_connect", [True, False])
def test_unet(model, in_channels, batch_norm, act_func, name, res_connect, res_connect_type, skip_connect):
    in_channels = 3
    input_shape = (128, 128, in_channels)
    hyperparameters = {
        "batch_normalization" : batch_norm,
        "activation" : act_func,
        "name" : name,
        "res_connect" : res_connect,
        "res_connect_type" : res_connect_type,
        "skip_connect" : skip_connect,
        "ratio" : 2 # only important for attention models
    }
    model_creation(model, in_channels, input_shape, hyperparameters)

def model_creation(model, in_channels, input_shape, hyperparameters={}, do_fit=False, do_plot=False):
    out_channels = 2
    batch = 4
    model_built: tf.keras.Model = model(
        tf.keras.Input(shape=input_shape, batch_size=batch, dtype=float),
        out_channels,
        "DICE",
        **hyperparameters
    )
    output_shape = model_built.output.shape.as_list()
    # make sure that the dimensions are right
    assert output_shape[0] == batch
    assert np.all(np.array(output_shape[1:-1]) == input_shape[:-1])
    assert output_shape[-1] == out_channels

    model_built.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics="acc",
        optimizer=tf.keras.optimizers.Adam(),
    )

    if do_fit:
        # create random data with simple thresholds as test data
        n_samples = batch * 2
        foreground = np.random.randint(2, size=(n_samples,) + input_shape[:-1] + (1,))
        background = 1 - foreground
        labels = np.concatenate([background, foreground], axis=-1)

        samples = np.random.normal(size=foreground.shape[:-1] + (in_channels,))
        # add label information
        for channel in range(in_channels):
            samples[..., channel] += labels[..., 1]

        model_built.fit(x=samples, y=labels, batch_size=batch)
    if do_plot:
        tf.keras.utils.plot_model(model_built, to_file=f"graph-{model.__name__}.png")


if __name__ == "__main__":

    # print all debug messages
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    for mod in [DeepLabv3plus, DenseTiramisu, unet, unet_old]:
        model_creation(mod, in_channels=3, input_shape=(128, 128, 3))
