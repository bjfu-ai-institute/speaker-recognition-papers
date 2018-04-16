import sys
sys.path.append("G:\\Cs\\github\\speaker-verification-papers\\DeepSpeaker")

import tensorflow as tf
import config

class Model(object):
    def __init__(self):
        # import setting from config
        self.n_speaker = config.N_SPEAKER
        self.n_blocks = config.N_RES_BLOCK
        self.n_gpu = config.N_GPU
        self.conv_weight_decay = config.CONV_WEIGHT_DECAY
        self.fc_weight_dacay = config.FC_WEIGHT_DECAY
        self.bn_epsilon = config.BN_EPSILON
        self.out_channel = config.OUT_CHANNEL
        
        self.build_graph()
        
    def build_graph(self):
        
        self.create_input()
        
        inp = self.batch_frames
        
        targets = self.batch_targets
        
        for i in range(self.n_blocks):
            if i > 0:
                inp = self.residual_block(inp,
                self.out_channel[i], "residual_block_%d"%i,
                is_first_layer=True)
        
            else:     
                inp = self.residual_block(inp,
                self.out_channel[i], "residual_block_%d"%i,
                is_first_layer=False)
        
        inp = tf.nn.avg_pool(inp, ksize=[1, 2, 2, 1], 
                             stride=[1, 1, 1, 1], padding='SAME')
        
        weight_affine = self.new_variable("affine_weight", [inp.get_shape[-1], 512],
                                          weight_type="FC")
        
        bias_affine = self.new_variable("affine_bias", [512], "FC")

        inp = tf.nn.relu(tf.matmul(inp, weight_affine) + bias_affine)

        output = self.batch_normalization(inp)

        self._vector = output

        self._loss = self.triplet_loss(output, targets)
    
    @property
    def loss(self):
        return self._loss

    def vector(self):
        return self._vector

    def create_input(self):
        self.batch_frames = tf.constant([None, 400, 400, 1])
        self.batch_targets = tf.constant([None, self.n_speaker])

    def sess_init(self):
        return

    def residual_block(self, inp, out_channel, name, is_first_layer=0):
        inp_channel = inp.get_shape().as_list()[-1]
        if inp_channel*2 == out_channel:
            increased = True
            stride = 2
        else:
            increased = False
            stride = 1
        if is_first_layer:
            weight = self.new_variable(name=name+"conv", shape=[3, 3, inp_channel, out_channel],
                                       weight_type="Conv")
            conv1 = tf.nn.conv2d(inp, weight, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = self.relu_conv_layer(inp, [3, 3, inp_channel, out_channel], name=name+"conv1",
                                         stride=stride, padding='SAME', bn_after_conv=False)
        conv2 = self.relu_conv_layer(conv1, [3, 3, out_channel, out_channel], name+"conv2",
                                     stride, 'SAME', bn_after_conv=False)
        if increased:
            pool_inp = tf.nn.avg_pool(inp, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
            padded_inp = tf.pad(pool_inp, [[0, 0], [0, 0], [0, 0], [inp_channel//2, inp_channel//2]])
        else:
            padded_inp = inp
        return conv2 + padded_inp

    def triplet_loss(self, inp, targets):
        loss = tf.contrib.triplet_semihard_loss(targets, inp, 1.0)
        return loss

    def batch_normalization(self, inp):
        dims = inp.get_shape()[-1]
        mean, variance = tf.nn.moments(inp, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dims, tf.float32,
                               initializer=tf.constant(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dims, tf.float32,
                                initializer=tf.constant(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(inp, mean, variance, beta, gamma, self.bn_epsilon)
        return bn_layer

    def relu_fc_layer(self, inp, units, name):
        weight_shape = [inp.get_shape()[-1], units]
        bias_shape = [units]
        weight = self.new_variable(name=name+"_weight", shape=weight_shape,
                                   weight_type="FC")
        bias = self.new_variable(name=name+"_bias", shape=bias_shape,
                                 weight_type="Conv")
        return tf.nn.relu(tf.matmul(inp, weight) + bias)

    def relu_conv_layer(self, inp, filter_shape, stride, padding,
                        name, bn_after_conv=False):
        weight = self.new_variable(name+"_filter", filter_shape, "Conv")
        if bn_after_conv:
            conv_layer = tf.nn.conv2d(inp, weight,
                                      strides=[1, stride, stride, 1], padding=padding)
            bn_layer = self.batch_normalization(conv_layer)
            output = tf.nn.relu(bn_layer)
            return output
        else:
            bn_layer = self.batch_normalization(inp)
            relu_layer = tf.nn.relu(bn_layer)
            conv_layer = tf.nn.conv2d(relu_layer, weight,
                                      strides=[1, stride, stride, 1], padding=padding)
            return conv_layer

    def new_variable(self, name, shape, weight_type, init=tf.contrib.layers.xavier_initializer()):
        if weight_type == "Conv":
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.conv_weight_decay)
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.fc_weight_dacay)
        new_var = tf.get_variable(name, shape=shape, initializer=init,
                                  regularizer=regularizer)
        return new_var
