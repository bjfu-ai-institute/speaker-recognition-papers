import os
import sys
import numpy as np
import tensorflow as tf
import src.loss.triplet_loss as triplet_loss
from src.data_manage import DataManage, DataManage4BigData

sys.path.append("..")


class DeepSpeaker:
    def __init__(self, config, out_channel=[64, 128, 256, 512]):
        self.n_blocks = len(out_channel)
        self.out_channel = out_channel
        self._n_speaker = config.N_SPEAKER
        self._embeddings = []
        self._max_step = config.MAX_STEP
        self._n_gpu = config.N_GPU
        self._conv_weight_decay = config.CONV_WEIGHT_DECAY
        self._fc_weight_decay = config.FC_WEIGHT_DECAY
        self._save_path = config.SAVE_PATH
        self._bn_epsilon = config.BN_EPSILON
        self._learning_rate = config.LR
        self._batch_size = config.BATCH_SIZE
        self._vectors = dict()

    @property
    def feature(self):
        return self._feature

    def _build_pred_graph(self):
        self._batch_frames = tf.placeholder(tf.float32, shape=[None, 100, 64, 1])
        self._batch_targets = tf.placeholder(tf.float32, shape=[None, 1])
        self._gpu_ind = tf.get_variable('gpu_ind', shape=[], dtype=tf.int32,
                                        initializer=tf.constant_initializer(value=0))
        raise NotImplementedError

    def _build_train_graph(self):
        self._batch_frames = tf.placeholder(tf.float32, shape=[self._n_gpu, None, 100, 64, 1])
        self._batch_targets = tf.placeholder(tf.float32, shape=[self._n_gpu, None, 1])
        self._gpu_ind = tf.get_variable('gpu_ind', shape=[], dtype=tf.int32,
                                        initializer=tf.constant_initializer(value=0))
        inp = self._batch_frames[self._gpu_ind]
        targets = tf.squeeze(self._batch_targets[self._gpu_ind])
        output = self._inference(inp)
        self._feature = output
        self._loss = self._triplet_loss(output, targets)
        self._opt = tf.train.AdamOptimizer(self._learning_rate)

    def _inference(self, inp):
        for i in range(self.n_blocks):
            if i > 0:
                inp = self._residual_block(inp, self.out_channel[i], "residual_block_%d" % i,
                                           is_first_layer=False)
        
            else:     
                inp = self._residual_block(inp, self.out_channel[i], "residual_block_%d" % i,
                                           is_first_layer=True)
        
        inp = tf.nn.avg_pool(inp, ksize=[1, 2, 2, 1], 
                             strides=[1, 1, 1, 1], padding='SAME')

        inp = tf.reshape(inp, [-1, inp.get_shape().as_list()[-1]])
        weight_affine = self._new_variable("affine_weight", [inp.get_shape().as_list()[-1], 512],
                                           weight_type="FC")
        
        bias_affine = self._new_variable("affine_bias", [512], "FC")

        inp = tf.nn.relu(tf.matmul(inp, weight_affine) + bias_affine)

        print(inp.get_shape().as_list())

        dims = inp.get_shape()[-1]
        mean, variance = tf.nn.moments(inp, axes=[0])
        beta = tf.get_variable('output_beta', dims, tf.float32,
                               initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable('output_gamma', dims, tf.float32,
                                initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.batch_normalization(inp, mean, variance, beta, gamma, self._bn_epsilon)
        
        return output

    @property
    def loss(self):
        return self._loss

    @property
    def vector(self):
        return self._vectors

    def _residual_block(self, inp, out_channel, name, is_first_layer=0):
        inp_channel = inp.get_shape().as_list()[-1]
        if inp_channel*2 == out_channel:
            increased = True
            stride = 2
        else:
            increased = False
            stride = 1
        if is_first_layer:
            weight = self._new_variable(name=name+"_conv", shape=[3, 3, inp_channel, out_channel],
                                        weight_type="Conv")
            conv1 = tf.nn.conv2d(inp, weight, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = self._relu_conv_layer(inp, [3, 3, inp_channel, out_channel], name=name+"_conv1",
                                          stride=stride, padding='SAME', bn_after_conv=False)
        conv2 = self._relu_conv_layer(conv1, [3, 3, out_channel, out_channel], name=name+"_conv2",
                                      stride=1, padding='SAME', bn_after_conv=False)
        if increased:
            pool_inp = tf.nn.avg_pool(inp, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')
            padded_inp = tf.pad(pool_inp, [[0, 0], [0, 0], [0, 0], [inp_channel//2, inp_channel//2]])
        else:
            padded_inp = inp
        return conv2 + padded_inp

    def _triplet_loss(self, inp, targets):
        loss = triplet_loss.batch_hard_triplet_loss(targets, inp, 1.0)
        return loss

    def _batch_normalization(self, inp, name):
        dims = inp.get_shape()[-1]
        mean, variance = tf.nn.moments(inp, axes=[0, 1, 2])
        beta = tf.get_variable(name+'_beta', dims, tf.float32,
                               initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable(name+'_gamma', dims, tf.float32,
                                initializer=tf.constant_initializer(value=0.0))
        bn_layer = tf.nn.batch_normalization(inp, mean, variance, beta, gamma, self._bn_epsilon)
        return bn_layer

    def _relu_fc_layer(self, inp, units, name):
        weight_shape = [inp.get_shape()[-1], units]
        bias_shape = [units]
        weight = self._new_variable(name=name+"_weight", shape=weight_shape,
                                    weight_type="FC")
        bias = self._new_variable(name=name+"_bias", shape=bias_shape,
                                  weight_type="Conv")
        return tf.nn.relu(tf.matmul(inp, weight) + bias)

    def _relu_conv_layer(self, inp, filter_shape, stride, padding,
                         name, bn_after_conv=False):
        weight = self._new_variable(name+"_filter", filter_shape, "Conv")
        if bn_after_conv:
            conv_layer = tf.nn.conv2d(inp, weight,
                                      strides=[1, stride, stride, 1], padding=padding)
            bn_layer = self._batch_normalization(conv_layer, name)
            output = tf.nn.relu(bn_layer)
            return output
        else:
            bn_layer = self._batch_normalization(inp, name)
            relu_layer = tf.nn.relu(bn_layer)
            conv_layer = tf.nn.conv2d(relu_layer, weight,
                                      strides=[1, stride, stride, 1], padding=padding)
            return conv_layer

    def _new_variable(self, name, shape, weight_type, init=tf.contrib.layers.xavier_initializer()):
        if weight_type == "Conv":
            regularizer = tf.contrib.layers.l2_regularizer(scale=self._conv_weight_decay)
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=self._fc_weight_decay)
        new_var = tf.get_variable(name, shape=shape, initializer=init,
                                  regularizer=regularizer)
        return new_var

    @staticmethod
    def _average_gradients(grads):  # grads:[[grad0, grad1,..], [grad0,grad1,..]..]
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

    def _validation_acc(self, enroll_frames, enroll_labels, test_frames, test_labels, sess):
        enroll_frames = np.array(enroll_frames[:200])
        enroll_labels = np.array(enroll_labels[:200])

        # if the labels aren't one-hot encode, convert them to one-hot

        enroll_f = np.zeros(shape=(self._n_gpu, enroll_frames.shape[0], 9, 40, 1))
        enroll_l = np.zeros(shape=(self._n_gpu, enroll_frames.shape[0], self._n_speaker))

        enroll_f[self._gpu_ind.eval()] = enroll_frames.reshape(-1, 9, 40, 1)
        enroll_l[self._gpu_ind.eval()] = enroll_labels.reshape(-1, self._n_speaker)

        features = sess.run(self.feature,
                            feed_dict={'x:0': enroll_f, 'y_:0': enroll_l})

        for i in range(self._n_speaker):
            self._vectors[i] = np.mean(features[np.max(enroll_labels) == i])

        test_f = np.zeros(shape=(self._n_gpu, test_frames.shape[0], 9, 40, 1))
        test_l = np.zeros(shape=(self._n_gpu, test_frames.shape[0], self._n_speaker))

        test_f[self._gpu_ind.eval()] = test_frames.reshape(-1, 9, 40, 1)
        test_l[self._gpu_ind.eval()] = test_labels.reshape(-1, self._n_speaker)

        features = sess.run(self.feature,
                            feed_dict={'x:0': test_f, 'y_:0': test_l})

        tot = 0
        acc = 0
        # print(features.shape[0])
        for vec_id in range(features.shape[0]):
            score = -1
            pred = -1
            tot += 1
            for spkr_id in range(self._n_speaker):
                if self._cosine(self._vectors[spkr_id], features[vec_id]) > score:
                    score = self._cosine(self._vectors[spkr_id], features[vec_id])
                    pred = spkr_id
            if pred == np.argmax(test_labels, axis=1)[vec_id]:
                acc += 1

        return acc / tot

    def _cosine(self, vector1, vector2):
        return np.dot(vector1, vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))

    def _train_step(self):
        grads = []
        for i in range(self._n_gpu):
            with tf.device("/gpu:%d" % i):
                self._gpu_ind.assign(i)
                gradient_all = self._opt.compute_gradients(self.loss)
                grads.append(gradient_all)
        with tf.device("/cpu:0"):
            ave_grads = self._average_gradients(grads)
            train_op = self._opt.apply_gradients(ave_grads)
        return train_op

    def run(self,
            train_frames, 
            train_labels,
            enroll_frames=None,
            enroll_labels=None,
            test_frames=None,
            test_labels=None):
        
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=False,
            )) as sess:
                self._build_train_graph()
                # Make sure the format of data is np.ndarray
                train_frames = np.array(train_frames)
                train_targets = np.array(train_labels)
                enroll_frames = np.array(enroll_frames)
                enroll_labels = np.array(enroll_labels)
                test_frames = np.array(test_frames)
                test_labels = np.array(test_labels)

                train_data = DataManage(train_frames, train_targets, self._batch_size)
                initial = tf.global_variables_initializer()
                sess.run(initial)
                saver = tf.train.Saver()
                train_op = self._train_step()

                # initial tensorboard
                writer = tf.summary.FileWriter(os.path.join(self._save_path, 'graph'), sess.graph)
                if enroll_frames is not None:
                    accuracy = self._validation_acc(sess, enroll_frames, enroll_labels, test_frames, test_labels)
                    acc_summary = tf.summary.scalar('accuracy', accuracy)
                # record the memory usage and time of each step
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                # define tensorboard steps

                merged_summary = tf.summary.merge_all()
                for i in range(self._max_step):
                    inp_frames = []
                    inp_labels = []
                    for i in range(self._n_gpu):
                        frames,labels = train_data.next_batch
                        inp_frames.append(frames)
                        inp_labels.append(labels)
                    inp_frames = np.array(inp_frames)
                    inp_labels = np.array(inp_labels)
                    sess.run(train_op, feed_dict={'x:0': inp_frames, 'y_:0': inp_labels})
                    if i % 25 == 0 or i+1 == self._max_step:
                        saver.save(sess, os.path.join(self._save_path, 'model'), global_step=i)

                INF = 0x3f3f3f3f
                self._n_gpu = 1
                enroll_data = DataManage(enroll_frames, enroll_labels, INF)
                test_data = DataManage(test_frames, test_labels, INF)

                get_vector = self.feature
                frames, labels = enroll_data.next_batch
                embeddings = sess.run(get_vector, feed_dict={'x:0': frames, 'y_:0': labels})

                for i in range(len(enroll_labels)):
                    if self._vectors[np.argmax(enroll_labels[i])]:
                        self._vectors[np.argmax(enroll_labels[i])] = embeddings[i]
                    else:
                        self._vectors[np.argmax(enroll_labels)[i]] += embeddings[i]
                        self._vectors[np.argmax(enroll_labels)[i]] /= 2
                
                frames, labels = test_data.next_batch
                embeddings = sess.run(get_vector, feed_dict={'x:0': frames, 'y_:0':labels})
                
                support = 0
                for i in range(len(embeddings)):
                    keys = self._vectors.keys()
                    score = 0
                    label = -1
                    for key in keys:
                        new_score = self._cosine(self._vectors[key], embeddings[i])
                        if new_score > score:
                            label = key
                    if label == np.argmax(test_labels[i]):
                        support += 1
                with open(os.path.join(self._save_path, 'result'), 'w') as f:
                    s = "Acc is %f" % (support/len(embeddings))
                    f.writelines(s)
