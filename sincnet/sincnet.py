import sys
sys.path.append('../')
import logging
from pyasv.basic import model
from pyasv import layers
from pyasv import Config
from pyasv import ops
from pyasv import utils
from pyasv import loss
import numpy as np
import h5py
from scipy.spatial.distance import cdist
import tensorflow as tf
import math
import os
import time


def to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def sinc_layer(x, out_channels, kernel_size, stride, padding, dilation, sample_rate, min_low_hz, min_band_hz):
    if type(x) == np.ndarray:
        if len(x.shape) == 2:
            dim = x.shape[-1]
            in_channel = 1
            x = x.reshape(-1, 1, dim)
        else:
            in_channel = x.shape[1]
    else:
        if len(x.get_shape()) == 2:
            in_channel = 1
            dim = x.get_shape()
            x = tf.reshape(x, shape=[-1, 1, dim])
        else:
            in_channel = x.get_shape().as_list()[1]
    if not kernel_size % 2:
        kernel_size += 1
    if in_channel != 1:
        raise ValueError("SincConv layer only support 1 channel")
    low_hz = 30
    high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
    mel = np.linspace(to_mel(low_hz),
                      to_mel(high_hz),
                      out_channels + 1)
    hz = to_hz(mel)
    low_hz_ = layers.new_variable(name="filter_low_hz", shape=[hz.shape[0]-1,],
                                  init=tf.constant_initializer(value=hz[:-1]))
    low_hz_ = tf.reshape(low_hz_, [-1, 1])

    band_hz_ = layers.new_variable(name="fliter_band_hz", shape=[hz.shape[0]-1,],
                                   init=tf.constant_initializer(value=np.diff(hz)))
    band_hz_ = tf.reshape(band_hz_, [-1, 1])

    n_lin = np.linspace(0, (kernel_size-1)/2-1, int(kernel_size/2))
    window = 0.54 - 0.46 * tf.math.cos(2 * math.pi * n_lin / kernel_size)

    n_ = 2 * math.pi * np.arange(-(kernel_size - 1)/2.0, 0).reshape(1, -1) / sample_rate

    low = min_low_hz + tf.abs(low_hz_)
    high = tf.clip_by_value(low + min_band_hz + tf.abs(band_hz_),
                            min_low_hz, sample_rate / 2)
    band = (high - low)[:, 0]
    f_times_t_low = tf.matmul(low, n_)
    f_times_t_high = tf.matmul(high, n_)

    band_pass_left = 2*((tf.math.sin(f_times_t_high) - tf.math.sin(f_times_t_low))/n_)*window
    band_pass_center = 2 * tf.reshape(band, [-1, 1])
    band_pass_right = tf.reverse(band_pass_left, axis=[1])

    filters = tf.reshape(tf.concat([band_pass_left, band_pass_center, band_pass_right], axis=1),
                         shape=[out_channels, 1, kernel_size])
    #[80, 1, 251] => [251, 1, 80]
    filters = tf.transpose(filters, [2, 0, 1])

    return tf.nn.conv1d(input=x, filters=filters, stride=stride)



class SincNet():
    pass