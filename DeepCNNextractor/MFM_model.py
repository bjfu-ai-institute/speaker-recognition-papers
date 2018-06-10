import tensorflow as tf
import sys
sys.path.append("..")
from scipy.spatial.distance import cosine
import os
import time
import config
import numpy as np

class DataManage(object):
    def __init__(self, raw_frames, raw_labels, batch_size):
        assert len(raw_frames) == len(raw_labels)
        # must be one-hot encoding
        self.raw_frames = np.array(raw_frames, dtype=np.float32)
        self.raw_labels = np.array(raw_labels, dtype=np.float32)
        self.batch_size = batch_size
        self.epoch_size = len(raw_frames) / batch_size
        self.batch_counter = 0
        self.is_eof = False
        self.spkr_num = np.array(self.raw_labels).shape[-1]

    @property
    def next_batch(self):
        if (self.batch_counter+1) * self.batch_size < len(self.raw_frames):
            batch_frames = self.raw_frames[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size]
            batch_labels = self.raw_labels[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size]
            self.batch_counter += 1
            return batch_frames, batch_labels
        else:
            self.is_eof = True
            return self.raw_frames[self.batch_counter * self.batch_size:-1], \
            self.raw_labels[self.batch_counter * self.batch_size:-1]

class DataManage4BigData(object):
    """
    Use this for huge dataset
    """
    def __init__(self, url):
        self.url = url
        self.batch_count = 0
        if os.path.exists(self.url) and os.listdir(self.url):
            self.file_is_exist = True
        else:
            self.file_is_exist = False

    def write_file(self, raw_frames, raw_labels):
        batch_size = config.BATCH_SIZE
        local_batch_count = 0
        if type(raw_frames) == np.ndarray:
            data_length = raw_frames.shape[0]
        else:
            data_length = len(raw_frames)
        while local_batch_count * batch_size < data_length:
            if local_batch_count * (batch_size+1) >= data_length:
                frames = raw_frames[local_batch_count * batch_size:]
                labels = raw_labels[local_batch_count * batch_size:]
                np.savez_compressed(os.path.join(self.url, "data_%d.npz"%local_batch_count), frames=frames, labels=labels)    
            else:
                frames = raw_frames[local_batch_count * batch_size: (local_batch_count+1) * batch_size]
                labels = raw_labels[local_batch_count * batch_size: (local_batch_count+1) * batch_size]
                np.savez_compressed(os.path.join(self.url, "data_%d.npz"%local_batch_count), frames=frames, labels=labels)    
                local_batch_count += 1

    @property
    def next_batch(self):
        if not self.file_is_exist:
            print('You need write file before load it.')
            exit()
        loaded = np.load(os.path.join(self.url, "data_%d.npz"%self.batch_count))
        frames = loaded['frames']
        labels = loaded['labels']
        self.batch_count += 1
        return frames, labels
        
class Model(object):
    def __init__(self):

        # import setting from ../config.py
        self.batch_size = config.BATCH_SIZE
        self.n_gpu = config.N_GPU
        self.name = config.MODEL_NAME
        self.n_speaker = config.N_SPEAKER
        self.max_step = config.MAX_STEP
        self.is_big_dataset = config.IS_BIG_DATASET
        if self.is_big_dataset:
            self.url_of_big_dataset = config.URL_OF_BIG_DATASET
        self.lr = config.LR
        self.save_path = config.SAVE_PATH
        
    @property
    def loss(self):
        return self._loss

    @property
    def prediction(self):
        return self._prediction

    @property
    def feature(self):
        return self._feature

    def build_pred_graph(self):
        pred_frames = tf.placeholder(tf.float32, shape=[None, 9, 40, 1], name='pred_x')
        frames = pred_frames
        _, self._feature = self.inference(frames)   

    def build_train_graph(self):
        self.gpu_ind = tf.get_variable(name='gpu_ind', trainable=False 
                            ,shape=[],dtype=tf.int32, initializer=tf.constant_initializer(value=0, dtype=tf.int32))
        frames = tf.placeholder(tf.float32, shape=[self.n_gpu, self.batch_size, 64], name='x')
        labels = tf.placeholder(tf.float32, shape=[self.n_gpu, self.batch_size, self.n_speaker], name='y_')
        out = self.inference(frames)
        out_softmax, feature = tf.nn.softmax(out)
        self._feature = feature
        self._prediction = out_softmax
        self._loss = -tf.reduce_mean(labels * tf.log(out_softmax))
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.opt.minimize(self._loss)
        self.saver = tf.train.Saver()

    def inference(self, frames):
        conv_1 = self.conv2d(frames, name='Conv1',shape=[7, 7, 1, 128], 
                            strides=[1, 1, 1, 1], padding='VALID')
        mfm_1 = self.max_feature_map(conv_1)
        pool_1 = tf.nn.max_pool(mfm_1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        conv_2_a = self.conv2d(pool_1, 'Conv2_a', shape=[1, 1, 64, 128], 
                               strides= [1, 1, 1, 1], padding='VALID')
        mfm_2_a = self.max_feature_map(conv_2_a)
        conv_2_b = self.conv2d(mfm_2_a, 'Conv2_b', shape=[5, 5, 64, 192], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_2_b = self.max_feature_map(conv_2_b)
        pool_2 = tf.nn.max_pool(mfm_2_b, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        conv_3_a = self.conv2d(pool_2, 'Conv3_a', shape=[1, 1, 96, 192], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_3_a = self.max_feature_map(conv_3_a)
        conv_3_b = self.conv2d(mfm_3_a, 'Conv3_b', shape=[5, 5, 96, 256], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_3_b = self.max_feature_map(conv_3_b)
        pool_3 = tf.nn.max_pool(mfm_3_b, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        conv_4_a = self.conv2d(pool_3, 'Conv4_a', shape=[1, 1, 128, 256], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_4_a = self.max_feature_map(conv_4_a)
        conv_4_b = self.conv2d(mfm_4_a, 'Conv4_b', shape=[3, 3, 128, 128], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_4_b = self.max_feature_map(conv_4_b)
        #

        conv_5_a = self.conv2d(mfm_4_b, 'Conv5_a', shape=[1, 1, 64, 128], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_5_a = self.max_feature_map(conv_5_a)
        conv_5_b = self.conv2d(mfm_5_a, 'Conv5_b', shape=[3, 3, 64, 128], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_5_b = self.max_feature_map(conv_2_b)
        pool_5 = tf.nn.max_pool(mfm_5_b, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        pool_5_flat = tf.reshape(pool_5, [-1,
                                        pool_5.get_shape().as_list()[1] *
                                        pool_5.get_shape().as_list()[2] *
                                        pool_5.get_shape().as_list()[3]])
        fc_1 = self.full_connect(pool_5_flat, name='fc_1', units=2048)
        mfm_6 = self.max_feature_map(fc_1, netType='fc')
        
        out = self.full_connect(mfm_6, name='out', units=self.n_speaker)
        return out, mfm_6

    def max_feature_map(self, x, netType='conv', name='activation'):
        if netType == 'fc':    
            x0, x1 = tf.split(x, num_or_size_splits = 2, axis = 1)
            y = tf.maximum(x0, x1)
        elif netType == 'conv':
            x0, x1 = tf.split(x, num_or_size_splits = 2, axis = 3) # split along the channel dimension
            y = tf.maximum(x0, x1)
        return y

    def t_dnn(self, x, shape, strides, name):
        with tf.name_scope(name):
            weights = self.weights_variable(shape)
        return tf.nn.conv1d(x, weights, stride=strides, padding='SAME', name=name+"_output")

    def conv2d(self, x, name, shape, strides, padding='SAME'):
        with tf.name_scope(name):
            weights = self.weights_variable(shape)
            biases = self.bias_variable(shape[-1])
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights,
                                                      strides=strides, padding=padding), biases, name=name+"_output"))

    def full_connect(self, x, name, units):
        with tf.name_scope(name):
            weights = self.weights_variable([x.get_shape().as_list()[-1], units])
            biases = self.bias_variable(units)
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), biases), name=name+"_output")

    @staticmethod
    def weights_variable(shape, name='weights', stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name='bias', stddev=0.1):
        initial = tf.truncated_normal([shape], stddev=stddev)
        return tf.Variable(initial, name=name)

    @staticmethod
    def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g,_ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
        return average_grads

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
            enroll_frames=None,
            enroll_label=None,
            test_frames=None,
            test_label=None,
            need_prediction_now=False):
        
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=False,
            )) as sess:
                if self.is_big_dataset:
                    train_data = DataManage4BigData(url = self.url_of_big_dataset)
                    if not train_data.file_is_exist:
                        train_data.write_file(train_frames, train_targets)
                    del train_frames, train_targets
                    self.build_train_graph()
                else:
                    self.build_train_graph()
                    train_data = DataManage(train_frames, train_targets, self.batch_size-1)
                initial = tf.global_variables_initializer()
                sess.run(initial)
                train_op = self.train_step()
                last_time = time.time()                
                for i in range(self.max_step):
                    input_frames = []
                    input_labels = []
                    for x in range(self.n_gpu):
                        frames, labels = train_data.next_batch
                        input_frames.append(frames)
                        input_labels.append(labels)
                    input_frames = np.array(input_frames).reshape([self.n_gpu, self.batch_size, 9, 40, 1])
                    input_labels = np.array(input_labels).reshape([self.n_gpu, self.batch_size, self.n_speaker])
                    sess.run(train_op, feed_dict={'x:0':input_frames, 'y_:0':input_labels})
                    current_time = time.time()
                    print("No.%d step use %f sec"%(i,current_time-last_time))
                    last_time = time.time()
                    if i % 10 == 0 or i + 1 ==self.max_step:
                        self.saver.save(sess, os.path.join(self.save_path,'model'))
        if need_prediction_now:
            self.run_predict(self.save_path, enroll_frames, enroll_label, test_frames, test_label)

    def run_predict(self, 
                    save_path,
                    enroll_frames,
                    enroll_targets, 
                    test_frames,
                    test_label):
        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                self.build_pred_graph()
                new_saver = tf.train.Saver()

                # needn't batch and gpu in prediction
                
                enroll_data = DataManage(enroll_frames, enroll_targets, self.batch_size)
                test_data = DataManage(test_frames, test_label, self.batch_size)
                new_saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
                feature_op = graph.get_operation_by_name('feature_layer_output')
                vector_dict = dict()
                while not enroll_data.is_eof:
                    frames, labels = enroll_data.next_batch
                    frames = np.array(frames).reshape([-1, 9, 40, 1])
                    labels = np.array(labels).reshape([-1, self.n_speaker])
                    vectors = sess.run(feature_op, feed_dict={'pred_x:0':frames})
                    for i in range(len(enroll_targets)):
                        if vector_dict[np.argmax(enroll_targets[i])]:
                            vector_dict[np.argmax(enroll_targets[i])] += vectors[i]
                            vector_dict[np.argmax(enroll_targets[i])] /= 2
                        else:
                            vector_dict[np.argmax(enroll_targets[i])] = vectors[i]
                while not test_data.is_eof:
                    frames, labels = test_data.next_batch
                    frames = np.array(frames).reshape([-1, 9, 40, 1])
                    labels = np.array(labels).reshape([-1, self.n_speaker])
                    vectors = sess.run(feature_op, feed_dict={'pred_x:0':frames})
                    keys = vector_dict.keys()
                    true_key = test_label
                    support = 0
                    for i in len(vectors):
                        score = 0
                        label = -1
                        for key in keys:
                            if cosine(vectors[i], vector_dict[key]) > score:
                                score = cosine(vectors[i], vector_dict[key])
                                label = key
                        if label == true_key[i]:
                            support += 1
                with open(os.path.join(save_path, 'log.txt'), 'w') as f:
                    s = "Acc = %f"%(support / test_data.raw_frames.shape[0])
                    f.writelines(s)
