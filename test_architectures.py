# pylint: disable=all

import logging
import sys

import numpy as np
import pytest
import tensorflow as tf

from .deeplab import DeepLabv3plus
from .densenets import DenseTiramisu
from .unets import unet

models = [DeepLabv3plus, DenseTiramisu, unet]


@pytest.mark.parametrize("model", models)
def test_model_creation(model):
    in_channels = 3
    out_channels = 2
    if model.__name__ == "DeepLabv3plus":
        input_shape = (256, 256, in_channels)
    else:
        input_shape = (128, 128, in_channels)
    batch = 4
    model_built: tf.keras.Model = model(
        tf.keras.Input(shape=input_shape, batch_size=batch, dtype=float),
        out_channels,
        "DICE",
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


if __name__ == "__main__":

    # print all debug messages
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    for mod in models:
        test_model_creation(mod)
