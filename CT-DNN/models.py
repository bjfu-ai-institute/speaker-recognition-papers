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
        self.raw_frames = np.array(raw_frames, dtype=np.float32)
        self.raw_labels = np.array(raw_labels, dtype=np.float32)
        self.batch_size = batch_size
        self.epoch_size = len(raw_frames) / batch_size
        self.enrollment_frames = np.array(enrollment_frames, dtype=np.float32)
        self.enrollment_targets = np.array(enrollment_targets, dtype=np.float32)
        self.batch_counter = 0
        self.spkr_num = np.array(self.raw_labels).shape[-1]

    def next_batch(self):
        if (self.batch_counter+1) * self.batch_size < len(self.raw_frames):
            batch_frames = self.raw_frames[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size+1]
            batch_labels = self.raw_labels[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size+1]
            self.batch_counter += 1

            return batch_frames, batch_labels
        else:
            spkr_num = len(self.raw_labels)
            return self.raw_frames, self.raw_labels, spkr_num

    @property
    def pred_data(self):
        return self.enrollment_frames, self.enrollment_targets, self.raw_frames, self.raw_labels


class Model(object):
    def __init__(self, n_gpu, max_step, sess, n_speaker,
                 save_path='/home/user1/fhq/ctdnn/save',
                 result_path='/home/user1/fhq/ctdnn/save'):

        self.sess = sess
        self.n_speaker = n_speaker
        self.result_path = result_path
        self.save_path = save_path
        self.max_step = max_step
        self.frames = tf.placeholder(tf.float32, [None, 9, 40, 1], name='x')
        self.target = tf.placeholder(tf.float32, [None, n_speaker], name='y')
        self.n_gpu = n_gpu
        self.prediction
        initializer = tf.global_variables_initializer()
        sess.run(initializer)

    @lazy_property
    def prediction(self):

        conv1 = self.conv2d(self.frames, 'conv1', [3, 11, 1, 128], [1, 1, 1, 1], 'SAME')

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')

        conv2 = self.conv2d(pool1, 'conv2', [3, 11, 128, 128], [1, 1, 1, 1], 'SAME')

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')

        pool2_flat = tf.reshape(pool2, [-1,
                                        pool2.get_shape().as_list()[1]*
                                        pool2.get_shape().as_list()[2]*
                                        pool2.get_shape().as_list()[3]])

        bottleneck = self.full_connect(pool2_flat, 'bottleneck', 512)

        bottleneck_t = tf.reshape(bottleneck, [-1, 512, 1])

        td1 = self.t_dnn(bottleneck_t, name='td1', shape=[2, 1, 128], strides=1)
        td1_flat = tf.reshape(td1, [-1, td1.get_shape().as_list()[1]*td1.get_shape().as_list()[2]])

        p_norm1 = self.full_connect(td1_flat, name='P_norm1', units=2000)
        p_norm1_t = tf.reshape(p_norm1, [-1, 2000, 1])

        td2 = self.t_dnn(p_norm1_t, name='td2', shape=[2, 1, 128], strides=1)
        td2_flat = tf.reshape(td2, [-1, td2.get_shape().as_list()[1]*td2.get_shape().as_list()[2]])

        p_norm2 = self.full_connect(td2_flat, name='P_norm2', units=400)

        feature_layer = self.full_connect(p_norm2, name='feature_layer', units=400)

        output = self.full_connect(feature_layer, name='output', units=self.n_speaker)

        return tf.nn.softmax(output), feature_layer

    @lazy_property
    def loss(self):
        logits = tf.placeholder(tf.float32, [None, self.n_speaker], name='logits')
        feature = tf.placeholder(tf.float32, [None, 400], name='feature')
        print("logits:", logits)
        true_target = self.target
        pred_id = tf.argmax(logits, axis=1)
        pred_id = tf.reshape(pred_id, [-1, 1])
        true_id = tf.argmax(true_target, axis=1)
        true_id = tf.reshape(true_id, [-1, 1])
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

    @staticmethod
    def train(sess, model, train_data, lr):
        grads = []
        for i in range(model.n_gpu):
            frames, ids = train_data.next_batch()
            with tf.device("/cpu:0"):
                print('predict')
                output, feature = sess.run(model.prediction, feed_dict={'x:0': frames})
                print("output:", output)
            with tf.device("/gpu:%d" % (i+3)):
                grad = sess.run(tf.gradients(model.loss, tf.trainable_variables()),
                                feed_dict={'logits:0': output, 'feature:0': feature, 'y:0': ids})
                print('loss')

                grads.append(grad)
        with tf.device("/cpu:0"):
            average_gradient = model.average_gradients(grads)
            opt = tf.train.AdamOptimizer(lr)
            train_op = opt.apply_gradients(average_gradient)
        return train_op, np.sum(grads)

    @staticmethod
    def run(train_data, predict_data, lr, n_gpu, max_step):
        assert type(train_data) == DataManage
        assert type(predict_data) == DataManage
        print("Enter succeed")
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
            )) as sess:
                model = Model(n_speaker=train_data.spkr_num, sess=sess, n_gpu=n_gpu, max_step=max_step)
                print("Init succeed")
                start_time = time.time()
                for step in range(model.max_step):
                    _, loss = sess.run(Model.train(sess, model, train_data, lr))
                    print("Running....")
                    saver = tf.train.Saver()
                    if not step % 10:
                        string = ("%d steps in %d sec. loss= %.2f" % step, time.time()-start_time, loss)
                        print(string)
                    if step % 1000 == 0 or step == model.max_step-1:
                        saver.save(sess, model.save_path)
                enroll_frames, enroll_target, test_frames, test_target = predict_data.pred_data
                _, feature = sess.run(model.prediction,
                                      feed_dict={'x:0': enroll_frames})
                true_id = np.argmax(enroll_target, axis=1)
                true_id = np.reshape(true_id, [-1, 1])
                vector = model.calc_vector(feature, true_id)
                _, feature = sess.run(model.prediction, feed_dict={'x:0': test_frames})
                predict_list = []
                for i in feature:
                    result = []
                    for vec in vector:
                        result.append(Model.cos_similar(vec, i))
                    if result[np.argmax(result)[0]] > 0.6:
                        predict_list.append(np.argmax(result))
                    else:
                        predict_list.append(model.n_speaker)
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

                with open(os.path.join(model.result_path, 'result.txt')) as file:
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
    def cos_similar(in_a, in_b):
        in_a = np.mat(in_a)
        in_b = np.mat(in_b)
        num = float(in_a * in_b.T)
        denom = np.linalg.norm(in_a) * np.linalg.norm(in_b)
        return 0.5 + 0.5 * (num / denom)

    @staticmethod
    def calc_vector(feature, targets):
        vectors = np.zeros([self.n_speaker, 400])
        for index, target in zip(range(targets.get_shape().as_list()[0]), targets):
            if vectors[target] == np.zeros(400):
                vectors[target] = feature[index]
            vectors[target] = np.add(vectors[index], feature[target])/2
        return vectors
