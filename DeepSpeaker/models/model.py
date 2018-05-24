import sys
sys.path.append("..")

import tensorflow as tf
import os
import numpy as np
import models.DataManage as DataManage
from scipy.spatial.distance import cosine
import triplet_loss
import config

class Model(object):
    def __init__(self):
        assert len(config.OUT_CHANNEL) == config.N_RES_BLOCKS, """
        assert len(config.OUT_CHANNEL) == config.N_RES_BLOCKS,
        OUT_CHANNEL is the array represents number of out channel of each residual block. 
        So the length of OUT_CHANNEL must equal to the N_RES_BLOCKS 
        """

        self.n_speaker = config.N_SPEAKER
        self.embeddings = []
        self.n_blocks = config.N_RES_BLOCKS
        self.max_step = config.MAX_STEP
        self.n_gpu = config.N_GPU
        self.conv_weight_decay = config.CONV_WEIGHT_DECAY
        self.fc_weight_dacay = config.FC_WEIGHT_DECAY
        self.save_path = config.SAVE_PATH
        self.bn_epsilon = config.BN_EPSILON
        self.out_channel = config.OUT_CHANNEL
        self.learning_rate = config.LEARNING_RATE
        self.batch_size = config.BATCH_SIZE
        self.build_graph()
        
    def build_graph(self):
        
        self.create_input()
        
        inp = self.batch_frames[self.gpu_ind]
        
        targets = tf.squeeze(self.batch_targets[self.gpu_ind])
        
        for i in range(self.n_blocks):
            if i > 0:
                inp = self.residual_block(inp,
                self.out_channel[i], "residual_block_%d"%i,
                is_first_layer=False)
        
            else:     
                inp = self.residual_block(inp,
                self.out_channel[i], "residual_block_%d"%i,
                is_first_layer=True)
        
        inp = tf.nn.avg_pool(inp, ksize=[1, 2, 2, 1], 
                             strides=[1, 1, 1, 1], padding='SAME')
        inp = tf.reshape(inp, [inp.get_shape().as_list()[0]*inp.get_shape().as_list()[1]*inp.get_shape().as_list()[2],
                                inp.get_shape().as_list()[-1]])
        weight_affine = self.new_variable("affine_weight", [inp.get_shape().as_list()[-1], 512],
                                          weight_type="FC")
        
        bias_affine = self.new_variable("affine_bias", [512], "FC")

        inp = tf.nn.relu(tf.matmul(inp, weight_affine) + bias_affine)

        print(inp.get_shape().as_list())

        dims = inp.get_shape()[-1]
        mean, variance = tf.nn.moments(inp, axes=[0])
        beta = tf.get_variable('output_beta', dims, tf.float32,
                               initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable('output_gamma', dims, tf.float32,
                                initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.batch_normalization(inp, mean, variance, beta, gamma, self.bn_epsilon)
        
        self._vector = output

        print(output.get_shape().as_list())
        print(targets.get_shape().as_list())
        self._loss = self.triplet_loss(output, targets)

        exit()

        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self._loss)

    @property
    def loss(self):
        return self._loss

    @property
    def vector(self):
        return self._vector

    def create_input(self):
        self.batch_frames = tf.placeholder(tf.float32, shape=[self.n_gpu, self.batch_size, 100, 64, 1])
        self.batch_targets = tf.placeholder(tf.float32, shape=[self.n_gpu, self.batch_size, 1])
        self.gpu_ind = tf.get_variable('gpu_ind', shape=[], dtype=tf.int32, 
                                       initializer=tf.constant_initializer(value=0))

    def residual_block(self, inp, out_channel, name, is_first_layer=0):
        print(name)
        print(inp.get_shape().as_list())
        inp_channel = inp.get_shape().as_list()[-1]
        if inp_channel*2 == out_channel:
            increased = True
            stride = 2
        else:
            increased = False
            stride = 1
        if is_first_layer:
            weight = self.new_variable(name=name+"_conv", shape=[3, 3, inp_channel, out_channel],
                                       weight_type="Conv")
            conv1 = tf.nn.conv2d(inp, weight, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = self.relu_conv_layer(inp, [3, 3, inp_channel, out_channel], name=name+"_conv1",
                                         stride=stride, padding='SAME', bn_after_conv=False)
        conv2 = self.relu_conv_layer(conv1, [3, 3, out_channel, out_channel], name=name+"_conv2",
                                     stride= 1, padding='SAME', bn_after_conv=False)
        if increased:
            pool_inp = tf.nn.avg_pool(inp, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')
            padded_inp = tf.pad(pool_inp, [[0, 0], [0, 0], [0, 0], [inp_channel//2, inp_channel//2]])
        else:
            padded_inp = inp
        return conv2 + padded_inp

    def triplet_loss(self, inp, targets):
        loss = triplet_loss.batch_all_triplet_loss(targets, inp, 1.0)
        return loss

    def batch_normalization(self, inp, name):
        dims = inp.get_shape()[-1]
        mean, variance = tf.nn.moments(inp, axes=[0, 1, 2])
        beta = tf.get_variable(name+'_beta', dims, tf.float32,
                               initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable(name+'_gamma', dims, tf.float32,
                                initializer=tf.constant_initializer(value=0.0))
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
            bn_layer = self.batch_normalization(conv_layer, name)
            output = tf.nn.relu(bn_layer)
            return output
        else:
            bn_layer = self.batch_normalization(inp, name)
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

    @staticmethod
    def average_gradients(grads):  # grads:[[grad0, grad1,..], [grad0,grad1,..]..]
        averaged_grads = []
        for grads_per_var in zip(*grads):
            grads = []
            for grad in grads_per_var:
                expanded_grad = tf.expand_dims(grad, 0)
                grads.append(expanded_grad)
            grads = tf.concat(grads, 0)
            grads = tf.reduce_mean(grads, 0)
            averaged_grads.append(grads)
        return averaged_grads

    def train_step(self):
        grads = []
        for i in range(self.n_gpu):
            with tf.device("/gpu:%d" % i):
                self.gpu_ind.assign(i)
                gradient_all = self.opt.compute_gradients(self.loss)
                grads.append(gradient_all)
        with tf.device("/cpu:0"):
            ave_grads = self.average_gradients(grads)
            train_op = self.opt.apply_gradients(ave_grads)
        return train_op

    def run(self,
            train_frames, 
            train_targets,
            enroll_frames,
            enroll_label,
            test_frames,
            test_label):
        
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=False,
            )) as sess:
                train_data = DataManage.DataManage(train_frames, train_targets, self.batch_size)
                initial = tf.global_variables_initializer()
                sess.run(initial)
                saver = tf.train.Saver()
                for i in range(self.max_step):
                    inp_frames = []
                    inp_labels = []
                    for i in range(self.n_gpu):
                        frames,labels = train_data.next_batch
                        inp_frames.append(frames)
                        inp_labels.append(labels)
                    train_op = self.train_step()
                    sess.run(train_op, feed_dict={'x:0':inp_frames, 'y_:0':inp_labels})
                    if i % 25 == 0 or i+1 == self.max_step:
                        saver.save(sess, os.path.join(self.save_path, 'model'), global_step=i)


                INF = 0x3f3f3f3f
                self.n_gpu = 1
                enroll_data = DataManage.DataManage(enroll_frames, enroll_label, INF)
                test_data = DataManage.DataManage(test_frames, test_label, INF)

                get_vector = self.vector                
                frames, labels = enroll_data.next_batch
                embeddings = sess.run(get_vector, feed_dict={'x:0':frames, 'y_:0':labels})

                self.vector_dict = dict()
                for i in range(len(enroll_label)):
                    if self.vector_dict[np.argmax(enroll_label[i])]:
                        self.vector_dict[np.argmax(enroll_label[i])] = embeddings[i]
                    else:
                        self.vector_dict[np.argmax(enroll_label)[i]] += embeddings[i]
                        self.vector_dict[np.argmax(enroll_label)[i]] /= 2
                
                frames, labels = test_data.next_batch
                embeddings = sess.run(get_vector, feed_dict={'x:0':frames, 'y_:0':labels})
                
                support = 0
                for i in range(len(embeddings)):
                    keys = self.vector_dict.keys()
                    score = 0
                    for key in keys:
                        new_score = cosine(self.vector_dict[key], embeddings[i])
                        if new_score > score:
                            label = key
                    if label == np.argmax(test_label[i]):
                        support += 1
                with open('/media/data/result/deep_speaker_in_c863', 'w') as f:
                    s = "Acc is %f" % (support/len(embeddings))
                    f.writelines(s)
