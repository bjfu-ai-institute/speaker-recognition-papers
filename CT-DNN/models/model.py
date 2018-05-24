import tensorflow as tf
import sys
sys.path.append("..")
from scipy.spatial.distance import cosine
import os
import time
import config
import numpy as np
import models.DataManage as DataManage


class Model:
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
        self.is_built = False
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

    def build_graph(self):
        """
        Build the compute graph.
        """
        self.gpu_ind = tf.get_variable(name='gpu_ind', trainable=False 
                            ,shape=[],dtype=tf.int32, initializer=tf.constant_initializer(value=0, dtype=tf.int32))
        self.batch_frames = tf.placeholder(tf.float32, shape=[self.n_gpu, self.batch_size, 9, 40, 1],name='x')
        self.batch_target = tf.placeholder(tf.float32, shape=[self.n_gpu, self.batch_size, self.n_speaker],name='y_')

        vector = np.zeros(shape=[self.n_speaker, 400])
        self.vectors = tf.Variable(vector, trainable=False, dtype=tf.float32)
            
        # Inference
        conv1 = self.conv2d(self.batch_frames[self.gpu_ind], 'conv1', [4, 8, 1, 128], [1, 1, 1, 1], 'VALID')

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 3, 1], strides=[1, 2, 3, 1], padding='VALID')

        conv2 = self.conv2d(pool1, 'conv2', [2, 4, 128, 128], [1, 1, 1, 1], 'VALID')

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

        pool2_flat = tf.reshape(pool2, [-1,
                                        pool2.get_shape().as_list()[1] *
                                        pool2.get_shape().as_list()[2] *
                                        pool2.get_shape().as_list()[3]])

        bottleneck = self.full_connect(pool2_flat, 'bottleneck', 512)

        bottleneck_t = tf.reshape(bottleneck, [-1, 512, 1])

        td1 = self.t_dnn(bottleneck_t, name='td1', shape=[5, 1, 128], strides=1)
        td1_flat = tf.reshape(td1, [-1, td1.get_shape().as_list()[1] * td1.get_shape().as_list()[2]])

        p_norm1 = self.full_connect(td1_flat, name='P_norm1', units=2000)
        p_norm1_t = tf.reshape(p_norm1, [-1, 2000, 1])

        td2 = self.t_dnn(p_norm1_t, name='td2', shape=[9, 1, 128], strides=1)
        td2_flat = tf.reshape(td2, [-1, td2.get_shape().as_list()[1] * td2.get_shape().as_list()[2]])

        p_norm2 = self.full_connect(td2_flat, name='P_norm2', units=400)

        feature_layer = self.full_connect(p_norm2, name='feature_layer', units=400)

        self._feature = feature_layer

        output = self.full_connect(feature_layer, name='output', units=self.n_speaker)

        self._prediction = tf.nn.softmax(output)

        self._loss = -tf.reduce_mean(self.batch_target[self.gpu_ind] * tf.log(output))
        """
        # Update vectors
        # pred_index = tf.argmax(output, axis=1)
        # true_index = tf.argmax(self.batch_target, axis=1)
        for i in range(self.n_speaker):
            vec_index = tf.where(tf.equal(true_index, i))
            vector = tf.reduce_mean(tf.gather(feature_layer, vec_index), axis=0)
            self.vectors[i].assign(vector)
            # tf.scatter_update(self.vectors, [i], vector)

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
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.opt.minimize(self._loss)
        self.saver = tf.train.Saver()
        self.is_built = True


    def t_dnn(self, x, shape, strides, name):
        with tf.name_scope(name):
            weights = self.weights_variable(shape)
            return tf.nn.conv1d(x, weights, stride=strides, padding='SAME')

    def conv2d(self, x, name, shape, strides, padding='SAME'):
        with tf.name_scope(name):
            weights = self.weights_variable(shape)
            biases = self.bias_variable(shape[-1])
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights,
                                                          strides=strides, padding=padding), biases))

    def full_connect(self, x, name, units):
        with tf.name_scope(name):
            weights = self.weights_variable([x.get_shape().as_list()[-1], units])
            biases = self.bias_variable(units)
            return tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), biases))

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

    def compute_exp_cosine(self, vector1, vector2=None):
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
            for i in range(self.n_speaker):
                vector = tf.gather(self.vectors, [i])
                vector = tf.expand_dims(vector, dim=1)
                x_norm = tf.sqrt(tf.reduce_sum(tf.square(vector1), axis=1))
                y_norm = tf.sqrt(tf.reduce_sum(tf.square(vector), axis=1))
                x_y = tf.reduce_sum(tf.multiply(vector1, vector), axis=1)
                cos_similarity = tf.divide(x_y, tf.multiply(x_norm, y_norm))
                tf.add(exp_sum, tf.exp(cos_similarity))
            return exp_sum

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
                    train_data = DataManage.DataManage4BigData(url = self.url_of_big_dataset)
                    if not train_data.file_is_exist:
                        train_data.write_file(train_frames, train_targets)
                    del train_frames, train_targets
                    self.build_graph()
                else:
                    self.build_graph()
                    train_data = DataManage.DataManage(train_frames, train_targets, self.batch_size-1)
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
                    input_frames = np.array(input_frames).reshape([4, self.batch_size, 9, 40, 1])
                    input_labels = np.array(input_labels).reshape([4, self.batch_size, self.n_speaker])
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
        with tf.Graph().as_default():
            with tf.Session() as sess:
                if not self.is_built:
                    self.build_graph()
                
                # needn't batch and gpu in prediction
                INF = 0x3f3f3f3f
                self.n_gpu = 1
                enroll_data = DataManage.DataManage(enroll_frames, enroll_targets, INF)
                test_data = DataManage.DataManage(test_frames, test_label, INF)
                new_saver = tf.train.import_meta_graph(os.path.join(self.save_path, 'model.meta'))
                new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(self.save_path, 'model.data-00000-of-00001')))
                feature_op = self.feature
                frames, labels = enroll_data.next_batch
                frames = np.array(frames).reshape([1, -1, 9, 40])
                labels = np.array(labels).reshape([1, -1, self.n_speaker])
                vectors = sess.run(feature_op, feed_dict={'x:0':frames, 
                                                        'y_:0':labels})
                vector_dict = dict()
                for i in len(enroll_targets):
                    if vector_dict[np.argmax(enroll_targets[i])]:
                        vector_dict[np.argmax(enroll_targets[i])] += vectors[i]
                        vector_dict[np.argmax(enroll_targets[i])] /= 2
                    else:
                        vector_dict[np.argmax(enroll_targets[i])] = vectors[i]
                frames, labels = test_data.next_batch
                frames = np.array(frames).reshape([1, -1, 9, 40])
                labels = np.array(labels).reshape([1, -1, self.n_speaker])
                vectors = sess.run(feature_op, feed_dict={'x:0':frames, 
                                                        'y_:0':labels})
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
                    s = "Acc = %f"%(support / len(vectors))
                    f.writelines(s)
