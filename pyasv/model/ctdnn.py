# -*- coding: utf-8 -*-


import tensorflow as tf
import sys
sys.path.append("..")
import os
import time
import numpy as np
from pyasv.data_manage import DataManage
from pyasv.data_manage import DataManage4BigData


class CTDnn:
    def __init__(self, config):
        """Create ctdnn model.

        Parameters
        ----------
        config : ``config`` class.
            The config of ctdnn model, no extra need.

        Notes
        -----
        The compute graph will not be created by ``__init__`` function.
        """
        self._config = config
        self._batch_size = config.BATCH_SIZE
        self._n_gpu = config.N_GPU
        self._name = config.MODEL_NAME
        self._n_speaker = config.N_SPEAKER
        self._max_step = config.MAX_STEP
        self._is_big_dataset = config.IS_BIG_DATASET
        if self._is_big_dataset:
            self._url_of_big_dataset = config.URL_OF_BIG_DATASET
        self._lr = config.LR
        self._save_path = config.SAVE_PATH
        self._vectors = np.zeros(shape=(self._n_speaker, 400))
        self._gpu_ind = 0

    def run(self,
            train_frames,
            train_labels,
            enroll_frames=None,
            enroll_labels=None,
            test_frames=None,
            test_labels=None,
            need_prediction_now=False):
        """Run the ctdnn model. Will save model to save_path/ and save tensorboard to save_path/graph/.

        Parameters
        ----------
        train_frames : ``list`` or ``np.ndarray``
            The feature array of train dataset.
        train_labels : ``list`` or ``np.ndarray``
            The label array of train dataset.
        enroll_frames : ``list`` or ``np.ndarray``
            The feature array of enroll dataset.
        enroll_labels=None ``list`` or ``np.ndarray``
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
                # convert all data to np.ndarray
                train_frames = np.array(train_frames)
                train_targets = np.array(train_labels)
                if train_targets.shape[-1] != self._n_speaker:
                    tmp = []
                    for i in range(train_targets.shape[0]):
                        tmp_line = np.zeros((self._n_speaker,))
                        tmp_line[np.argmax(train_targets[i])] = 1
                        tmp.append(tmp_line)
                    train_targets = np.array(tmp)
                else:
                    train_targets = np.array(train_targets)

                if enroll_frames is not None:
                    enroll_frames = np.array(enroll_frames)
                if enroll_labels is not None:
                    enroll_labels = np.array(enroll_labels)
                    if enroll_labels.shape[-1] != self._n_speaker:
                        tmp = []
                        for i in range(enroll_labels.shape[0]):
                            tmp_line = np.zeros((self._n_speaker,))
                            tmp_line[np.argmax(enroll_labels[i])] = 1
                            tmp.append(tmp_line)
                        enroll_labels = np.array(tmp)
                    else:
                        enroll_labels = np.array(test_labels)

                if test_frames is not None:
                    test_frames = np.array(test_frames)
                if test_labels is not None:
                    test_labels = np.array(test_labels)
                    if test_labels.shape[-1] != self._n_speaker:
                        tmp = []
                        for i in range(test_labels.shape[0]):
                            tmp_line = np.zeros((self._n_speaker,))
                            tmp_line[np.argmax(test_labels[i])] = 1
                            tmp.append(tmp_line)
                        test_labels = np.array(tmp)
                    else:
                        test_labels = np.array(test_labels)

                # initial tensorboard
                writer = tf.summary.FileWriter(os.path.join(self._save_path, 'graph'), sess.graph)

                # prepare data
                if self._is_big_dataset:
                    train_data = DataManage4BigData(self._config)
                    if not train_data.file_is_exist:
                        train_data.write_file(train_frames, train_targets)
                    del train_frames, train_targets
                    self._build_train_graph()
                else:
                    self._build_train_graph()
                    train_data = DataManage(train_frames, train_targets, self._config)

                # initial step
                initial = tf.global_variables_initializer()
                sess.run(initial)
                train_op, loss = self._train_step()
                if enroll_frames is not None:
                    accuracy = self._validation_acc(sess, enroll_frames, enroll_labels, test_frames, test_labels)
                    acc_summary = tf.summary.scalar('accuracy', accuracy)
                # record the memory usage and time of each step
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                # define tensorboard steps

                loss_summary = tf.summary.scalar('loss_summary', loss)
                merged_summary = tf.summary.merge_all()

                last_time = time.time()

                # train loop
                for i in range(self._max_step):
                    # get data
                    input_frames = []
                    input_labels = []
                    for x in range(self._n_gpu):
                        frames, labels = train_data.next_batch
                        L = []
                        for m in range(labels.shape[0]):
                            ids = np.zeros(self._n_speaker)
                            ids[np.argmax(enroll_labels[m])] = 1
                            L.append(ids)
                        labels = L
                        input_frames.append(frames)
                        input_labels.append(labels)
                    input_frames = np.array(input_frames).reshape([self._n_gpu, -1, 9, 40, 1])
                    input_labels = np.array(input_labels).reshape([self._n_gpu, -1, self._n_speaker])

                    _, summary_str = sess.run([train_op, merged_summary],
                                              feed_dict={'x:0': input_frames, 'y_:0': input_labels})
                    current_time = time.time()

                    # print log
                    print("-----------------------")
                    print("No.%d step use %f sec" % (i, current_time - last_time))
                    if enroll_frames is not None:
                        print("Acc = %f"%sess.run(accuracy))
                    print("-----------------------")
                    last_time = time.time()

                    # record
                    if i % 10 == 0 or i + 1 == self._max_step:
                        self._saver.save(sess, os.path.join(self._save_path, 'model'))

                    writer.add_run_metadata(run_metadata, 'step%d' % i)
                    writer.add_summary(summary_str, i)

        if need_prediction_now:
            self.run_predict(enroll_frames, enroll_labels, test_frames, test_labels)
        writer.close()

    def run_predict(self,
                    enroll_frames,
                    enroll_labels,
                    test_frames,
                    test_labels):
        """Run prediction, will save the result to save_path/result.txt

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
                    L = []
                    for i in range(labels.shape[0]):
                        ids = np.zeros(self._n_speaker)
                        ids[np.max(labels[i])] = 1
                        L.append(ids)
                    labels = L
                    frames = np.array(frames).reshape([-1, 9, 40, 1])
                    labels = np.array(labels).reshape([-1, self._n_speaker])
                    vectors = sess.run(feature_op, feed_dict={'pred_x:0': frames})
                    for i in range(len(enroll_labels)):
                        if vector_dict[np.argmax(enroll_labels[i])]:
                            vector_dict[np.argmax(enroll_labels[i])] += vectors[i]
                            vector_dict[np.argmax(enroll_labels[i])] /= 2
                        else:
                            vector_dict[np.argmax(enroll_labels[i])] = vectors[i]

                support = 0
                while not test_data.is_eof:
                    frames, labels = test_data.next_batch
                    L = []
                    for i in range(labels.shape[0]):
                        ids = np.zeros(self._n_speaker)
                        ids[np.max(labels[i])] = 1
                        L.append(ids)
                    labels = L
                    frames = np.array(frames).reshape([-1, 9, 40, 1])
                    labels = np.array(labels).reshape([-1, self._n_speaker])
                    vectors = sess.run(feature_op, feed_dict={'pred_x:0': frames})
                    keys = vector_dict.keys()
                    true_key = test_labels
                    for i in len(vectors):
                        score = 0
                        label = -1
                        for key in keys:
                            if self._cosine(vectors[i], vector_dict[key]) > score:
                                score = self._cosine(vectors[i], vector_dict[key])
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
        prediction: ``tf.operation``
        """
        return self._prediction

    @property
    def feature(self):
        """A ``property`` member to get feature.

        Returns
        -------
        feature: ``tf.operation``
        """
        return self._feature

    def _build_pred_graph(self):
        pred_frames = tf.placeholder(tf.float32, shape=[None, 9, 40, 1], name='pred_x')
        frames = pred_frames
        _, self._feature = self._inference(frames)   

    def _build_train_graph(self):

        batch_frames = tf.placeholder(tf.float32, shape=[None, 9, 40, 1],name='x')
        batch_target = tf.placeholder(tf.float32, shape=[None, self._n_speaker], name='y_')
        frames = batch_frames[self._batch_size*self._gpu_ind:(self._batch_size+1)*self._gpu_ind, :, :, :]
        target = batch_target[self._batch_size*self._gpu_ind:(self._batch_size+1)*self._gpu_ind, :]
        self._prediction, self._feature = self._inference(frames)
        self._loss = -tf.reduce_mean(tf.reduce_sum(target*tf.log(tf.clip_by_value(self._prediction,
                                                                                  1e-29, 1.0)), axis=1))
        """
        # Update vectors
        # pred_index = tf.argmax(output, axis=1)
        # true_index = tf.argmax(self.batch_target, axis=1)
        for i in range(self.n_speaker):
            vec_index = tf.where(tf.equal(true_index, i))
            vector = tf.reduce_mean(tf.gather(feature_layer, vec_index), axis=0)
            self.vectors[i].assign(vector)
            # tf.scatter_update(    vectors, [i], vector)

        # Compute loss
        loss = tf.Variable(0, dtype=tf.float32)
        for i in range(pred_index.get_shape().as_list()[0]):
            vec_ind = tf.gather(pred_index, i)
            pred_vector = tf.gather(self.vectors, vec_ind)
            vec_ind = tf.gather(true_index, i)
            true_vector = tf.gather(self.vectors, vec_ind)
            distance_1 = self.compute_exp_cosine(pred_vector, true_vector)
            distance_2 = self.compute_exp_cosine(pred_vector)
            if loss:
                loss += tf.negative(tf.log(tf.divide(distance_1, distance_2)))
            else:
                loss = tf.negative(tf.log(tf.divide(distance_1, distance_2)))
        """
        self._opt = tf.train.AdamOptimizer(self._lr)
        self._opt.minimize(self._loss)
        self._saver = tf.train.Saver()
        
    def _inference(self, frames):
            
        # Inference
        conv1 = self._conv2d(frames, 'conv1', [4, 8, 1, 128], [1, 1, 1, 1], 'VALID')

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 3, 1], strides=[1, 2, 3, 1], padding='VALID')

        conv2 = self._conv2d(pool1, 'conv2', [2, 4, 128, 128], [1, 1, 1, 1], 'VALID')

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

        pool2_flat = tf.reshape(pool2, [-1,
                                        pool2.get_shape().as_list()[1] *
                                        pool2.get_shape().as_list()[2] *
                                        pool2.get_shape().as_list()[3]])

        bottleneck = self._full_connect(pool2_flat, 'bottleneck', 512)

        bottleneck_t = tf.reshape(bottleneck, [-1, 512, 1])

        td1 = self._t_dnn(bottleneck_t, name='td1', shape=[5, 1, 128], strides=1)
        td1_flat = tf.reshape(td1, [-1, td1.get_shape().as_list()[1] * td1.get_shape().as_list()[2]])

        p_norm1 = self._full_connect(td1_flat, name='P_norm1', units=2000)
        p_norm1_t = tf.reshape(p_norm1, [-1, 2000, 1])

        td2 = self._t_dnn(p_norm1_t, name='td2', shape=[9, 1, 128], strides=1)
        td2_flat = tf.reshape(td2, [-1, td2.get_shape().as_list()[1] * td2.get_shape().as_list()[2]])

        p_norm2 = self._full_connect(td2_flat, name='P_norm2', units=400)

        feature_layer = self._full_connect(p_norm2, name='feature_layer', units=400)

        output = self._full_connect(feature_layer, name='output', units=self._n_speaker)

        return output, feature_layer
        
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

    def _compute_exp_cosine(self, vector1, vector2=None):
        exp_sum = tf.Variable(0, trainable=False, dtype=np.float32)
        vector1 = tf.expand_dims(vector1, dim=1)
        if vector2 is not None:
            vector2 = tf.expand_dims(vector2, dim=1)
            x_norm = tf.sqrt(tf.reduce_sum(tf.square(vector1), axis=1))
            y_norm = tf.sqrt(tf.reduce_sum(tf.square(vector2), axis=1))
            x_y = tf.reduce_sum(tf.multiply(vector1, vector2), axis=1)
            cos_similarity = tf.divide(x_y, tf.multiply(x_norm, y_norm))
            tf.add(exp_sum, tf.exp(cos_similarity))
            return exp_sum
        else:
            for i in range(self._n_speaker):
                vector = tf.gather(self._vectors, [i])
                vector = tf.expand_dims(vector, dim=1)
                x_norm = tf.sqrt(tf.reduce_sum(tf.square(vector1), axis=1))
                y_norm = tf.sqrt(tf.reduce_sum(tf.square(vector), axis=1))
                x_y = tf.reduce_sum(tf.multiply(vector1, vector), axis=1)
                cos_similarity = tf.divide(x_y, tf.multiply(x_norm, y_norm))
                tf.add(exp_sum, tf.exp(cos_similarity))
            return exp_sum

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
            loss = tf.reduce_sum(self.loss)
        return train_op, loss

    def _validation_acc(self, sess, enroll_frames, enroll_labels, test_frames, test_labels):
        enroll_frames = np.array(enroll_frames[:200])
        enroll_labels = np.array(enroll_labels[:200])

        # if the labels aren't one-hot encode, convert them to one-hot
        
        enroll_f = np.zeros(shape=(self._n_gpu, enroll_frames.shape[0], 9, 40, 1))
        enroll_l = np.zeros(shape=(self._n_gpu, enroll_frames.shape[0], self._n_speaker))

        enroll_f[self._gpu_ind.eval()] = enroll_frames.reshape(-1, 9, 40 ,1)
        enroll_l[self._gpu_ind.eval()] = enroll_labels.reshape(-1, self._n_speaker)

        features = sess.run(self.feature, 
                            feed_dict={'x:0':enroll_f, 'y_:0':enroll_l})
        
        for i in range(self._n_speaker):
            self._vectors[i] = np.mean(features[np.max(enroll_labels) == i])

        test_f = np.zeros(shape=(self._n_gpu, test_frames.shape[0], 9, 40, 1))
        test_l = np.zeros(shape=(self._n_gpu, test_frames.shape[0], self._n_speaker))

        test_f[self._gpu_ind.eval()] = test_frames.reshape(-1, 9, 40 ,1)
        test_l[self._gpu_ind.eval()] = test_labels.reshape(-1, self._n_speaker)

        features = sess.run(self.feature, 
                            feed_dict={'x:0':test_f, 'y_:0':test_l})
        
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
        return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))

