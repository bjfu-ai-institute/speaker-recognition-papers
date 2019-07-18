import sys
sys.path.append('../..')
import numpy as np
from pyasv.basic import model
from pyasv.basic import layers
from pyasv.config import Config
import tensorflow as tf


class XVector(model.Model):
    def __init__(self, config):
        super().__init__(config)
        pass


    def inference(self, x):
        with tf.variable_scope('Forward', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('T_DNN_1', reuse=tf.AUTO_REUSE):
                out_1 = layers.t_dnn(x, out=512, length=2, strides=1, name='',
                             init=tf.contrib.layers.xavier_initializer)
            with tf.variable_scope('T_DNN_2', reuse=tf.AUTO_REUSE):
                out_2 = layers.t_dnn(out_1, length=2, strides=1, name='',
                                     init=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('T_DNN_3', reuse=tf.AUTO_REUSE):
                out_3 = layers.t_dnn(out_2, length=3, strides=1, name='',
                                     init=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('T_DNN_4', reuse=tf.AUTO_REUSE):
                out_4 = layers.t_dnn(out_3, length=1, strides=1, name='',
                                     init=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('T_DNN_5', reuse=tf.AUTO_REUSE):
                out_5 = layers.t_dnn(out_4, length=1, strides=1, name='',
                                     init=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('Pooling', reuse=tf.AUTO_REUSE):
                out_6 = layers.static_pooling(out_5)
            with tf.variable_scope('FC_1', reuse=tf.AUTO_REUSE):
                out_7 = layers.full_connect(out_6, name="", units=512)
            with tf.variable_scope('FC_2', reuse=tf.AUTO_REUSE):
                out_8 = layers.full_connect(out_7, name="", units=512)
            with tf.variable_scope('Out', reuse=tf.AUTO_REUSE):
                out_9 = layers.full_connect(out_8, name="", units=self.config.n_speaker)
        return out_9

    def loss(self, y_, y):
        return tf.nn.softmax_cross_entropy_with_logits_v2(y, y_)
