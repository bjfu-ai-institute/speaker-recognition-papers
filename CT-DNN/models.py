import tensorflow as tf
import time
import os
import numpy as np
import functools


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class DataManage(object):
    def __init__(self, raw_frames, raw_labels, batch_size,
                 enrollment_frames=None, enrollment_targets=None):
        assert len(raw_frames) == len(raw_labels)
        # must be one-hot encoding
        assert len(np.array(raw_labels).shape) == 2

        self.raw_frames = raw_frames
        self.raw_labels = raw_labels
        self.batch_size = batch_size
        self.epoch_size = len(raw_frames) / batch_size
        self.enrollment_frames = enrollment_frames
        self.enrollment_targets = enrollment_targets
        self.batch_counter = 0

    def next_batch(self):
        if (self.batch_counter+1) * self.batch_size < len(self.raw_frames):
            batch_frames = self.raw_frames[self.batch_counter * self.batch_size,
                                           (self.batch_counter+1) * self.batch_size]
            batch_labels = self.raw_labels[self.batch_counter * self.batch_size,
                                           (self.batch_counter+1) * self.batch_size]
            self.batch_counter += 1
            spkr_num = np.array(batch_labels).shape[1]
            return batch_frames, batch_labels, spkr_num
        else:
            spkr_num = np.array(self.raw_labels).shape[1]
            return self.raw_frames, self.raw_labels, spkr_num

    @property
    def pred_data(self):
        return self.enrollment_frames, self.enrollment_targets, self.raw_frames, self.raw_labels


class Model:
    def __init__(self, n_gpu, max_step,
                 save_path='/home/user1/fhq/ctdnn/save',
                 result_path='/home/user1/fhq/ctdnn/save'):

        self.n_speaker = None
        self.result_path = result_path
        self.save_path = save_path
        self.max_step = max_step
        self.batch_frames = None
        self.batch_target = None
        self.n_gpu = n_gpu

    @lazy_property
    def prediction(self):
        n_speaker = self.n_speaker
        conv1 = self.conv2d(self.batch_frames, 'conv1', [6, 33, 1, 128], [1, 2, 3, 1], 'SAME')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 11, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = self.conv2d(pool1, 'conv2', [6, 33, 1, 128], [1, 2, 3, 1], 'SAME')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 11, 1], strides=[1, 2, 2, 1], padding='SAME')
        bottleneck = self.full_connection(pool1, 'bottleneck', [512])
        td1 = self.t_dnn(bottleneck, 'td1', [512], [2], [1])
        p_norm1 = self.full_connection(td1, 'P_norm1', [2000])
        td2 = self.t_dnn(p_norm1, 'td2', [2000], [2], [1])
        p_norm2 = self.full_connection(td2, 'P_norm2', [400])
        feature_layer = self.full_connection(p_norm2, 'feature_layer', [400])
        pre_output = self.full_connection(feature_layer, 'output', [n_speaker])
        return tf.nn.softmax(pre_output), feature_layer

    @lazy_property
    def loss(self):
        logits, feature = self.prediction
        logits = np.array(logits)
        pred_id = np.argmax(logits)
        true_id = np.argmax(self.batch_target)
        vector = self.calc_vector(feature, true_id)
        loss = []
        for vec_pred, vec_true in zip(vector[pred_id], vector[true_id]):
            exp_all = 0
            for index in range(self.n_speaker):
                exp_all += tf.exp(self.cos_similar(vector[index], vec_pred))
            exp_pred = tf.exp(self.cos_similar(vec_pred, vec_true))
            loss.append(-exp_pred/exp_all)
        loss = np.sum(np.array(loss))
        return loss

    def train(self, x, y, n_speaker, lr):
        grads = []
        for i in range(self.n_gpu):
            with tf.device("/gpu:%d" % i):
                self.batch_frames = x
                self.batch_target = y
                self.n_speaker = n_speaker
                grad = tf.gradients(self.loss, tf.trainable_variables())
                grads.append(grad)
        with tf.device("/cpu:0"):
            average_gradient = self.average_gradients(grads)
            opt = tf.train.AdamOptimizer(lr)
            train_op = opt.apply_gradients(average_gradient)
        return train_op

    def run(self, train_data, predict_data, lr):
        assert type(train_data) == DataManage
        assert type(predict_data) == DataManage
        with tf.Graph().as_default():
            initializer = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(initializer)
                start_time = time.time()
                for step in range(self.max_step):
                    x, y, n_speaker = train_data.next_batch()
                    _, loss = sess.run([self.train(x, y, n_speaker, lr), ])
                    saver = tf.train.Saver()
                    if not step % 10:
                        string = ("%d steps in %d sec. loss= %.2f" % step, time.time()-start_time, loss)
                        print(string)
                    if step % 1000 == 0 or step == self.max_step-1:
                        saver.save(sess, self.save_path)
                self.batch_frames, self.batch_target, x, y = predict_data.pred_data
                _, feature = self.prediction
                true_id = np.argmax(self.batch_target)
                spkr_num = np.array(self.batch_target).shape[1]
                vector = self.calc_vector(feature, true_id)
                self.batch_frames = x
                self.batch_target = y
                _, feature = self.prediction
                predict_list = []
                for i in feature:
                    result = []
                    for vec in vector:
                        result.append(self.cos_similar(vec, i))
                    if result[np.argmax(result)[0]] > 0.6:
                        predict_list.append(np.argmax(result))
                    else:
                        predict_list.append(spkr_num)
                support = 0
                false_reject = 0
                false_accept = 0
                for i in range(len(feature)):
                    if predict_list[i] == y[i]:
                        support += 1
                    else:
                        if predict_list[i] == spkr_num and y[i] != spkr_num:
                            false_reject += 1
                        elif predict_list[i] != spkr_num and y[i] == spkr_num:
                            false_accept += 1

                with open(os.path.join(self.result_path, 'result.txt')) as file:
                    file.write(("ACC= %d, EER= %d" % (support/len(feature), false_reject/false_accept)))

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

    def conv2d(self, x, name, shape, strides, padding='SAME'):
        with tf.name_scope(name):
            weights = self.weights_variable(shape)
            biases = self.bias_variable(shape)
            return tf.nn.relu(tf.nn.conv2d(x, weights, strides=strides, padding=padding) + biases)

    def full_connection(self, x, name, shape, activation=True):
        with tf.name_scope(name):
            weights = self.weights_variable(shape)
            biases = self.bias_variable(shape[-1])
            if activation:
                return tf.nn.relu(tf.matmul(x, weights) + biases)
            else:
                return tf.matmul(x, weights) + biases

    def t_dnn(self, x, name, filters, kernel, stride, padding='SAME'):
        with tf.name_scope(name):
            weights = self.weights_variable(kernel)
            biases = self.bias_variable(filters)
            return tf.nn.relu(tf.nn.conv1d(x, weights, stride=stride, padding=padding) + biases)

    @staticmethod
    def weights_variable(shape, name='weights', stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name='bias', stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    @staticmethod
    def cos_similar(in_a, in_b):
        in_a = np.mat(in_a)
        in_b = np.mat(in_b)
        num = float(in_a * in_b.T)
        denom = np.linalg.norm(in_a) * np.linalg.norm(in_b)
        return 0.5 + 0.5 * (num / denom)

    @staticmethod
    def calc_vector(feature, targets):
        vectors = np.zeros([len(targets), 400])
        for index, target in zip(range(len(targets)), targets):
            if vectors[target] == np.zeros(400):
                vectors[target] = feature[index]
            vectors[target] = np.add(vectors[index], feature[target])/2
        return vectors
