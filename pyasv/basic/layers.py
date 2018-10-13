# -*- coding: utf-8 -*-
#


import tensorflow as tf


def new_variable(name, shape, weight_decay=0.001,
                 init=tf.contrib.layers.xavier_initializer()):
    with tf.device('/cpu:0'):
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        new_var = tf.get_variable(name, shape=shape, initializer=init,
                                  regularizer=regularizer)
    return new_var


def t_dnn(x, shape, strides, name):
    weights = new_variable(shape=shape, name=name+'_w')
    return tf.nn.conv1d(x, weights, stride=strides, padding='SAME', name=name + "_output")


def conv2d(x, name, shape, strides, padding, weight_decay=0.001):
    weights = new_variable(shape=shape, name=name+'_w', weight_decay=weight_decay)
    biases = new_variable(shape=[shape[-1]], name=name+'_b', weight_decay=weight_decay)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights,
                                                  strides=strides, padding=padding), biases, name=name + "_output"))


def batch_normalization(inp, name, epsilon):
    bn_layer = tf.layers.batch_normalization(inp, name=name, epsilon=epsilon)
    return bn_layer


def full_connect(x, name, units):
    weights = new_variable(shape=[x.get_shape().as_list()[-1], units], name=name+'w')
    biases = new_variable(shape=units, name=name+'b')
    return tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), biases), name=name+"_output")


def lstm(x, units, is_training, layers, forget_bias=1.0, output_keep_prob=0.5):
    batch_size, seq_len, _ = x.get_shape().as_list()
    stack_rnn = []
    for i in range(layers):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=units, forget_bias=forget_bias, state_is_tuple=True)
        if is_training:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
        stack_rnn.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell=cell, input=x, initial_state=initial_state, dtype=tf.float32)
    return outputs, states


def Blstm():
    print("Not implemented now.")
    pass


def _max_feature_map(x, netType='conv'):
    if netType == 'fc':
        x0, x1 = tf.split(x, num_or_size_splits=2, axis=1)
        y = tf.maximum(x0, x1)
    elif netType == 'conv':
        x0, x1 = tf.split(x, num_or_size_splits=2, axis=3)  # split along the channel dimension
        y = tf.maximum(x0, x1)
    return y
