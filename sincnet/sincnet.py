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


def sinc_layer(x, out_channels, kernel_size, stride, sample_rate, min_low_hz, min_band_hz):
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

    # After simplification.
    band_pass_left = 2*((tf.math.sin(f_times_t_high) - tf.math.sin(f_times_t_low))/n_)*window
    band_pass_center = 2 * tf.reshape(band, [-1, 1])
    band_pass_right = tf.reverse(band_pass_left, axis=[1])

    filters = tf.reshape(tf.concat([band_pass_left, band_pass_center, band_pass_right], axis=1),
                         shape=[out_channels, 1, kernel_size])
    #[80, 1, 251] => [251, 1, 80]
    filters = tf.transpose(filters, [2, 0, 1])

    return tf.nn.conv1d(input=x, filters=filters, stride=stride)


class SincNet_ID(model.Model):
    def __init__(self, config, out_channel, kernel_size, is_training):
        super().__init__(config)
        self.ks = kernel_size
        self.out = out_channel

    def inference(self, x):
        with tf.variable_scope('sinc', reuse=tf.AUTO_REUSE):
            out = layers.layer_norm(x, 'ln_inp')
            out = sinc_layer(out, out_channels=self.out, kernel_size=self.ks,
                             stride=1, sample_rate=self.config.sample_rate,
                             min_low_hz=30, min_band_hz=50)
            out = layers.layer_norm(out, 'ln')
        with tf.variable_scope('conv1d_1', reuse=tf.AUTO_REUSE):
            out = layers.t_dnn(out, length=3, strides=1, out=60, name='c1')
            out = layers.layer_norm(out, 'ln')
        with tf.variable_scope('conv1d_2', reuse=tf.AUTO_REUSE):
            out = layers.t_dnn(out, length=3, strides=1, out=60, name='c1')
            out = layers.layer_norm(out, 'ln')
        with tf.variable_scope('fc_1', reuse=tf.AUTO_REUSE):
            out = layers.full_connect(out, name='fc1', units=2048, activation='leakyrelu')
            out = layers.layer_norm(out, 'ln')

        with tf.variable_scope('fc_2', reuse=tf.AUTO_REUSE):
            out = layers.full_connect(out, name='fc2', units=2048, activation='leakyrelu')
            out = layers.layer_norm(out, 'ln')
        with tf.variable_scope('fc_3', reuse=tf.AUTO_REUSE):
            out = layers.full_connect(out, name='fc3', units=2048, activation='leakyrelu')
            out = layers.layer_norm(out, 'ln')
        with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
            out = layers.full_connect(out, name='out', units=self.config.n_speaker)
            # pre softmax, calc loss with tf.cross_entr..... and predict with tf.nn.softmax
        return tf.nn.softmax(out)
