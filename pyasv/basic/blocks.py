import tensorflow as tf
from pyasv.basic import layers


def residual_block(inp, out_channel, name, weight_decay=0.001, epsilon=0.001, is_first_layer=False):
    inp_channel = inp.get_shape().as_list()[-1]
    if inp_channel*2 == out_channel:
        increased = True
        stride = 2
    else:
        increased = False
        stride = 1
    if is_first_layer:
        weight = layers.new_variable(name=name+"_conv", shape=[3, 3, inp_channel, out_channel],
                                     weight_decay=weight_decay)
        conv1 = tf.nn.conv2d(inp, weight, strides=[1, 1, 1, 1], padding="SAME")
    else:
        conv1 = relu_conv_layer(inp, [3, 3, inp_channel, out_channel], name=name+"_conv1",
                                stride=stride, padding='SAME', bn_after_conv=False, epsilon=epsilon)
    conv2 = relu_conv_layer(conv1, [3, 3, out_channel, out_channel], name=name+"_conv2",
                            stride=1, padding='SAME', bn_after_conv=False, epsilon=epsilon)
    if increased:
        padded_inp = tf.pad(inp, [[0, 0], [0, 0], [0, 0], [inp_channel//2, inp_channel//2]])
    else:
        padded_inp = inp
    return conv2 + padded_inp


def relu_conv_layer(inp, filter_shape, stride, padding,
                    epsilon, name, bn_after_conv=False):
    weight = layers.new_variable(name+"_filter", filter_shape)
    if bn_after_conv:
        conv_layer = tf.nn.conv2d(inp, weight,
                                  strides=[1, stride, stride, 1], padding=padding)
        bn_layer = layers.batch_normalization(conv_layer, name, epsilon=epsilon)
        output = tf.nn.relu(bn_layer)
        return output
    else:
        bn_layer = layers.batch_normalization(inp, name, epsilon=epsilon)
        relu_layer = tf.nn.relu(bn_layer)
        conv_layer = tf.nn.conv2d(relu_layer, weight,
                                  strides=[1, stride, stride, 1], padding=padding)
        return conv_layer
