import tensorflow as tf
import sys
sys.path.append("..")
from scipy.spatial.distance import cosine
import os
import time
import numpy as np
from pyasv.data_manage import DataManage
from pyasv.data_manage import DataManage4BigData


class MaxFeatureMapDnn:
    def __init__(self, config):
        """Create Max feature Map model.

        Parameters
        ----------
        config : ``config`` class.
            The config of MFM model, no extra need.

        Notes
        -----
        The compute graph will not be created by ``__init__`` function.

        """
        self._config = config
        self._batch_size = config.BATCH_SIZE
        self._vectors = []
        self._n_gpu = config.N_GPU
        self._name = config.MODEL_NAME
        self._n_speaker = config.N_SPEAKER
        self._gpu_ind = 0
        self._max_step = config.MAX_STEP
        self._is_big_dataset = config.IS_BIG_DATASET
        if self._is_big_dataset:
            self._url_of_big_dataset = config.URL_OF_BIG_DATASET
        self._lr = config.LR
        self._save_path = config.SAVE_PATH

    def run(self,
            train_frames,
            train_labels,
            enroll_frames=None,
            enroll_labels=None,
            test_frames=None,
            test_labels=None,
            need_prediction_now=False):
        """Run the MFM model. Will save model to save_path/ and save tensorboard to save_path/graph/.

        Parameters
        ----------
        train_frames : ``list`` or ``np.ndarray``
            The feature array of train dataset.
        train_labels : ``list`` or ``np.ndarray``
            The label array of train dataset.
        enroll_frames : ``list`` or ``np.ndarray``
            The feature array of enroll dataset.
        enroll_labels : ``list`` or ``np.ndarray``
            The label array of enroll dataset.
        test_frames : ``list`` or ``np.ndarray``
            The feature array of test dataset.
        test_labels : ``list`` or ``np.ndarray``
            The label array of test dataset.
        need_prediction_now : ``bool``
            if *True* we will create predict graph and run predict now.
            if *False* we will exit after training.
        """

        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=False,
            )) as sess:
                if self._is_big_dataset:
                    train_data = DataManage4BigData(self._config)
                    if not train_data.file_is_exist:
                        train_data.write_file(train_frames, train_labels)
                    del train_frames, train_labels
                    self._build_train_graph()
                else:
                    self._build_train_graph()
                    train_data = DataManage(train_frames, train_labels, self._batch_size - 1)
                initial = tf.global_variables_initializer()
                sess.run(initial)
                train_op = self._train_step()
                last_time = time.time()
                for i in range(self._max_step):
                    input_frames = []
                    input_labels = []
                    for x in range(self._n_gpu):
                        print(x)
                        frames, labels = train_data.next_batch
                        input_frames.append(frames)
                        input_labels.append(labels)
                    input_frames = np.array(input_frames).reshape([-1, 9, 40, 1])
                    input_labels = np.array(input_labels).reshape([-1, self._n_speaker])
                    sess.run(train_op, feed_dict={'x:0': input_frames, 'y_:0': input_labels})
                    current_time = time.time()
                    print("No.%d step use %f sec" % (i, current_time - last_time))
                    last_time = time.time()
                    if i % 10 == 0 or i + 1 == self._max_step:
                        self._saver.save(sess, os.path.join(self._save_path, 'model'))
        if need_prediction_now:
            self.run_predict(enroll_frames, enroll_labels, test_frames, test_labels)

    def run_predict(self,
                    enroll_frames,
                    enroll_labels,
                    test_frames,
                    test_labels):
        """Run prediction, will save the result to save_path

        Parameters
        ----------
        enroll_frames : ``list`` or ``np.ndarray``
            The feature array of enroll dataset.
        enroll_labels : ``list`` or ``np.ndarray``
            The label of enrol dataset.
        test_frames : ``list`` or ``np.ndarray``
            The feature array of test dataset.
        test_labels :
            The label of test dataset.
        """
        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                self._build_pred_graph()
                new_saver = tf.train.Saver()

                # needn't batch and gpu in prediction

                enroll_data = DataManage(enroll_frames, enroll_labels, self._batch_size)
                test_data = DataManage(test_frames, test_labels, self._batch_size)
                new_saver.restore(sess, tf.train.latest_checkpoint(self._save_path))
                feature_op = graph.get_operation_by_name('feature_layer_output')
                vector_dict = dict()
                while not enroll_data.is_eof:
                    frames, labels = enroll_data.next_batch
                    frames = np.array(frames).reshape([-1, 9, 40, 1])
                    labels = np.array(labels).reshape([-1, self._n_speaker])
                    vectors = sess.run(feature_op, feed_dict={'pred_x:0': frames})
                    for i in range(len(enroll_labels)):
                        if vector_dict[np.argmax(enroll_labels[i])]:
                            vector_dict[np.argmax(enroll_labels[i])] += vectors[i]
                            vector_dict[np.argmax(enroll_labels[i])] /= 2
                        else:
                            vector_dict[np.argmax(enroll_labels[i])] = vectors[i]
                while not test_data.is_eof:
                    frames, labels = test_data.next_batch
                    frames = np.array(frames).reshape([-1, 9, 40, 1])
                    labels = np.array(labels).reshape([-1, self._n_speaker])
                    vectors = sess.run(feature_op, feed_dict={'pred_x:0': frames})
                    keys = vector_dict.keys()
                    true_key = test_labels
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
                with open(os.path.join(self._save_path, 'result.txt'), 'w') as f:
                    s = "Acc = %f" % (support / test_data.raw_frames.shape[0])
                    f.writelines(s)

    @property
    def loss(self):
        """A ``property`` member to get loss.

        Returns
        -------
        loss : ``tf.operation``
        """
        return self._loss

    @property
    def prediction(self):
        """A ``property`` member to do predict operation.

        Returns
        -------
        prediction : ``tf.operation``
        """
        return self._prediction

    @property
    def feature(self):
        """A ``property`` member to get feature.

        Returns
        -------
        feature : ``tf.operation``
        """
        return self._feature

    def _build_pred_graph(self):
        pred_frames = tf.placeholder(tf.float32, shape=[None, 9, 40, 1], name='pred_x')
        frames = pred_frames
        _, self._feature = self._inference(frames)   

    def _build_train_graph(self):
        frames = tf.placeholder(tf.float32, shape=[-1, 64, 1], name='x')
        labels = tf.placeholder(tf.float32, shape=[-1, self._n_speaker], name='y_')
        inp = frames[self._gpu_ind*self._batch_size:(self._gpu_ind+1)*self._batch_size, :, :, :]
        label = labels[self._gpu_ind*self._batch_size:(self._gpu_ind+1)*self._batch_size, :, :, :]
        out, mfm6 = self._inference(frames)
        self._feature = out
        self._prediction = out
        self._loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=out)
        self._opt = tf.train.AdamOptimizer(self._lr)
        self._opt.minimize(self._loss)
        self._saver = tf.train.Saver()

    def _inference(self, frames):
        conv_1 = self._conv2d(frames, name='Conv1',shape=[7, 7, 1, 128], 
                            strides=[1, 1, 1, 1], padding='VALID')
        mfm_1 = self._max_feature_map(conv_1)
        pool_1 = tf.nn.max_pool(mfm_1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        conv_2_a = self._conv2d(pool_1, 'Conv2_a', shape=[1, 1, 64, 128], 
                               strides= [1, 1, 1, 1], padding='VALID')
                               
        mfm_2_a = self._max_feature_map(conv_2_a)
        conv_2_b = self._conv2d(mfm_2_a, 'Conv2_b', shape=[5, 5, 64, 192], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_2_b = self._max_feature_map(conv_2_b)
        pool_2 = tf.nn.max_pool(mfm_2_b, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        conv_3_a = self._conv2d(pool_2, 'Conv3_a', shape=[1, 1, 96, 192], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_3_a = self._max_feature_map(conv_3_a)
        conv_3_b = self._conv2d(mfm_3_a, 'Conv3_b', shape=[5, 5, 96, 256], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_3_b = self._max_feature_map(conv_3_b)
        pool_3 = tf.nn.max_pool(mfm_3_b, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        conv_4_a = self._conv2d(pool_3, 'Conv4_a', shape=[1, 1, 128, 256], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_4_a = self._max_feature_map(conv_4_a)
        conv_4_b = self._conv2d(mfm_4_a, 'Conv4_b', shape=[3, 3, 128, 128], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_4_b = self._max_feature_map(conv_4_b)
        #

        conv_5_a = self._conv2d(mfm_4_b, 'Conv5_a', shape=[1, 1, 64, 128], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_5_a = self._max_feature_map(conv_5_a)
        conv_5_b = self._conv2d(mfm_5_a, 'Conv5_b', shape=[3, 3, 64, 128], 
                               strides=[1, 1, 1, 1], padding='VALID')
        mfm_5_b = self._max_feature_map(conv_5_b)
        pool_5 = tf.nn.max_pool(mfm_5_b, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        pool_5_flat = tf.reshape(pool_5, [-1,
                                        pool_5.get_shape().as_list()[1] *
                                        pool_5.get_shape().as_list()[2] *
                                        pool_5.get_shape().as_list()[3]])
        fc_1 = self._full_connect(pool_5_flat, name='fc_1', units=2048)
        mfm_6 = self._max_feature_map(fc_1, netType='fc')
        
        out = self._full_connect(mfm_6, name='out', units=self._n_speaker)
        return out, mfm_6

    def _max_feature_map(self, x, netType='conv', name='activation'):
        if netType == 'fc':    
            x0, x1 = tf.split(x, num_or_size_splits = 2, axis = 1)
            y = tf.maximum(x0, x1)
        elif netType == 'conv':
            x0, x1 = tf.split(x, num_or_size_splits = 2, axis = 3) # split along the channel dimension
            y = tf.maximum(x0, x1)
        return y

    def _t_dnn(self, x, shape, strides, name):
        with tf.name_scope(name):
            weights = self._weights_variable(shape)
        return tf.nn.conv1d(x, weights, stride=strides, padding='SAME', name=name+"_output")

    def _conv2d(self, x, name, shape, strides, padding='SAME'):
        with tf.name_scope(name):
            weights = self._weights_variable(shape)
            biases = self._bias_variable(shape[-1])
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights,
                                                      strides=strides, padding=padding), biases, name=name+"_output"))

    def _full_connect(self, x, name, units):
        with tf.name_scope(name):
            weights = self._weights_variable([x.get_shape().as_list()[-1], units])
            biases = self._bias_variable(units)
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), biases), name=name+"_output")

    @staticmethod
    def _weights_variable(shape, name='weights', stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    @staticmethod
    def _bias_variable(shape, name='bias', stddev=0.1):
        initial = tf.truncated_normal([shape], stddev=stddev)
        return tf.Variable(initial, name=name)

    @staticmethod
    def _average_gradients(tower_grads):
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

    def _validation_acc(self, sess, enroll_frames, enroll_labels, test_frames, test_labels):
        enroll_frames = np.array(enroll_frames[:200])
        enroll_labels = np.array(enroll_labels[:200])
        test_labels = np.array(test_labels)

        features = sess.run(self.feature,
                            feed_dict={'x:0': enroll_frames, 'y_:0': enroll_labels})

        for i in range(self._n_speaker):
            self._vectors[i] = np.mean(features[np.max(enroll_labels) == i])

        features = sess.run(self.feature,
                            feed_dict={'x:0': test_frames, 'y_:0': test_labels})

        tot = 0
        acc = 0
        # print(features.shape[0])
        for vec_id in range(test_labels.shape[0]):
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
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))

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
