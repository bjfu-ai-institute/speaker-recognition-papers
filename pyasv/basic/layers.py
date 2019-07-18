# -*- coding: utf-8 -*-
#


import tensorflow as tf


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


def t_dnn(x, length, strides, name, out=None, padding='VALID', init=tf.constant_initializer(value=1)):
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
    else:
        return "activation param should be one of [relu, tanh, softmax, None] now."


def lstm(x, units, is_training, layers, batch_size, forget_bias=1.0, output_keep_prob=0.5):
    shapes = x.get_shape().as_list()
    _, seq_len, _ = shapes[-3], shapes[-2], shapes[-1]
    stack_rnn = []
    for i in range(layers):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=units, forget_bias=forget_bias, state_is_tuple=True)
        if is_training:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
        stack_rnn.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32)
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
        sequence_length=sequence_length,  #[FRAMES_PER_SAMPLE] * self.batch_size
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


def static_pooling(x):
    dim = len(x.get_shape().as_list())
    if dim > 3 and x.shape[-1] != 1:
        raise ValueError("Input for static pooling should be 3-d tensor, [batch_size, time_series, out_channel]")
    mean, var = tf.nn.moments(x, axes=1)
    return tf.concat([mean, tf.sqrt(var)], axis=1)