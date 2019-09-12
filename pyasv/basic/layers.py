# -*- coding: utf-8 -*-
#


import tensorflow as tf
import numpy as np
import math


def new_variable(name, shape, weight_decay=0.001,
                 init=tf.contrib.layers.xavier_initializer(), forced_gpu=None):
    if forced_gpu is not None:
        with tf.device('/gpu:%d'%forced_gpu):
            regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
            new_var = tf.get_variable(name, shape=shape, initializer=init,
                                      regularizer=regularizer)
    else:
        with tf.device('/cpu:0'):
            regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
            new_var = tf.get_variable(name, shape=shape, initializer=init,
                                      regularizer=regularizer)
    return new_var


def t_dnn(x, length, strides, name, out=None, padding='VALID', init=tf.contrib.layers.xavier_initializer()):
    dimension = len(x.get_shape().as_list())
    if out is None:
        out = x.shape[-1]
    if dimension == 3:
        pass
    elif x.get_shape().as_list()[-1] == 1 and dimension == 4:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    else:
        raise ValueError("Input of T_DNN should be 3-d tensor or 4-d tensor like [x, y, z, 1], now got %d-d"%dimension)
    weights = new_variable(init=init, shape=[length, x.shape[-1], out], name=name+'_w')
    return tf.nn.conv1d(x, weights, stride=strides, padding=padding, name=name + "_output")


def conv2d(x, name, shape, strides, padding, weight_decay=0.001):
    weights = new_variable(shape=shape, name=name+'_w', weight_decay=weight_decay)
    biases = new_variable(shape=[shape[-1]], name=name+'_b', weight_decay=weight_decay)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights,
                                                  strides=strides, padding=padding), biases, name=name + "_output"))


def batch_normalization(inp, name, epsilon):
    bn_layer = tf.layers.batch_normalization(inp, name=name, epsilon=epsilon)
    return bn_layer


def full_connect(x, name, units, activation='relu'):
    weights = new_variable(shape=[x.get_shape().as_list()[-1], units], name=name+'w')
    biases = new_variable(shape=units, name=name+'b')
    if activation == 'relu':
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), biases), name=name+"_output")
    elif activation is None:
        return tf.nn.bias_add(tf.matmul(x, weights), biases)
    elif activation == 'tanh':
        return tf.nn.tanh(tf.nn.bias_add(tf.matmul(x, weights), biases))
    elif activation == 'softmax':
        return tf.nn.softmax(tf.nn.bias_add(tf.matmul(x, weights), biases))
    elif activation == 'None':
        return tf.nn.bias_add(tf.matmul(x, weights), biases)
    elif activation == 'leakyrelu':
        return tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(x, weights), biases))
    else:
        return "activation param should be one of [relu, tanh, softmax, None] now."


def lstm(x, units, is_training, layers, forget_bias=1.0, output_keep_prob=0.9):
    shapes = x.get_shape().as_list()
    _, seq_len, _ = shapes[-3], shapes[-2], shapes[-1]
    stack_rnn = []
    for i in range(layers):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=units, forget_bias=forget_bias, state_is_tuple=True)
        if is_training:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
        stack_rnn.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, time_major=True)
    return outputs, states


def Blstm(x, n_hidden, sequence_length, input_keep_prob=1, output_keep_prob=1, dropout_keep_prob=1, layer_norm=False):
    lstm_fw_cell = tf.contrib.rnn.LaryerNormBasicLSTMCell(
        n_hidden, layer_norm=layer_norm,
        dropout_keep_prob=dropout_keep_prob)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_fw_cell, input_keep_prob=input_keep_prob,
        output_keep_prob=output_keep_prob)
    lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        n_hidden, layer_norm=layer_norm,
        dropout_keep_prob=dropout_keep_prob)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_bw_cell, input_keep_prob=input_keep_prob,
        output_keep_prob=output_keep_prob)
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        lstm_fw_cell, lstm_bw_cell, x,
        sequence_length=sequence_length,  # [FRAMES_PER_SAMPLE] * self.batch_size
        dtype=tf.float32)
    return outputs, state


def _max_feature_map(x, netType='conv'):
    if netType == 'fc':
        x0, x1 = tf.split(x, num_or_size_splits=2, axis=1)
        y = tf.maximum(x0, x1)
    elif netType == 'conv':
        x0, x1 = tf.split(x, num_or_size_splits=2, axis=3)  # split along the channel dimension
        y = tf.maximum(x0, x1)
    else:
        raise TypeError("net Type should be `conv` or `fc`")
    return y


def layer_norm(x, name, eps=1e-6):
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, axis=-1)
        gamma = new_variable('gamma', shape=[], init=tf.constant_initializer(value=1.0))
        beta = new_variable('beta', shape=[], init=tf.constant_initializer(value=0.0))
    return gamma * (x - mean) / (tf.sqrt(var) + eps) + beta


def static_pooling(x):
    dim = len(x.get_shape().as_list())
    if dim > 3 and x.shape[-1] != 1:
        raise ValueError("Input for static pooling should be 3-d tensor, [batch_size, time_series, out_channel]")
    mean, var = tf.nn.moments(x, axes=1)
    return tf.concat([mean, tf.sqrt(var)], axis=1)


def sinc_layer(x, out_channels, kernel_size, stride, sample_rate, min_low_hz, min_band_hz):
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

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
    low_hz_ = new_variable(name="filter_low_hz", shape=[hz.shape[0]-1,],
                                  init=tf.constant_initializer(value=hz[:-1]))
    low_hz_ = tf.reshape(low_hz_, [-1, 1])

    band_hz_ = new_variable(name="fliter_band_hz", shape=[hz.shape[0]-1,],
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