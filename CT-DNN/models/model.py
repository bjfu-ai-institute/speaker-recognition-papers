import tensorflow as tf
import sys
sys.path.append("..")
import config
import numpy as np
import DataManage


class Model:
    def __init__(self,
                 n_gpu,
                 n_speaker,
                 model_name):

        self.n_speaker = n_speaker
        self.name = model_name
        self.batch_frames = tf.constant(value=0.0, shape=[1, 9, 40, 1], dtype=tf.float32)
        self.batch_target = tf.constant(value=0.0, shape=[1, n_speaker], dtype=tf.float32)
        self.n_gpu = n_gpu
        self._prediction = None
        vector = np.zeros(shape=[n_speaker, 400])
        self.vectors = tf.Variable(vector, trainable=False, dtype=tf.float32)
        self._feature = None
        self._loss = None

        with tf.variable_scope(model_name):
            self.build_graph()

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

        # Inference
        conv1 = self.conv2d(self.batch_frames, 'conv1', [4, 8, 1, 128], [1, 1, 1, 1], 'VALID')

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

        # Update vectors
        pred_index = tf.argmax(output, axis=1)
        true_index = tf.argmax(self.batch_target, axis=1)
        for i in range(self.n_speaker):
            vec_index = tf.where(tf.equal(true_index, i))
            vector = tf.reduce_mean(tf.gather_nd(feature_layer, vec_index), axis=0)
            self.vectors[i].assign(vector)

        # Compute loss
        loss = tf.Variable(0, dtype=tf.float32)
        for i in range(pred_index.get_shape().as_list()[0]):
            vec_ind = tf.gather_nd(pred_index, [i])
            pred_vector = tf.gather_nd(self.vectors, [vec_ind])
            vec_ind = tf.gather_nd(true_index, [i])
            true_vector = tf.gather_nd(self.vectors, [vec_ind])
            distance_1 = self.compute_exp_cosine(pred_vector, true_vector)
            distance_2 = self.compute_exp_cosine(pred_vector)
            loss = tf.add(tf.negative(tf.log(tf.divide(distance_1, distance_2))), loss)
        self._loss = loss

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
                vector = tf.gather_nd(self.vectors, [i])
                vector = tf.expand_dims(vector, dim=1)
                x_norm = tf.sqrt(tf.reduce_sum(tf.square(vector1), axis=1))
                y_norm = tf.sqrt(tf.reduce_sum(tf.square(vector), axis=1))
                x_y = tf.reduce_sum(tf.multiply(vector1, vector), axis=1)
                cos_similarity = tf.divide(x_y, tf.multiply(x_norm, y_norm))
                tf.add(exp_sum, tf.exp(cos_similarity))
            return exp_sum

    def train_step(self, train_data, lr):
        assert type(train_data) == DataManage
        grads = []
        opt = tf.train.AdamOptimizer(lr)
        for i in range(self.n_gpu):
            with tf.device("/gpu:%d" % i):
                frames, targets = train_data.next_batch()
                frames = tf.constant(frames, dtype=tf.float32)
                targets = tf.constant(targets, dtype=tf.float32)
                self.batch_frames = frames
                self.batch_target = targets
                gradient_all = opt.compute_gradients(self.loss)
                grads.append(gradient_all)
        with tf.device("/cpu:0"):
            ave_grads = self.average_gradients(grads)
            train_op = opt.apply_gradients(ave_grads)
        return train_op, tf.reduce_sum(grads)


def run(train_frames,
        train_targets,
        batch_size,
        max_step,
        save_path,
        n_gpu,
        lr):
    lr = config.LR
    n_gpu = config.N_GPU
    save_path = config.SAVE_PATH
    max_step = config.MAX_STEP
    batch_size = config.BATCH_SIZE
    
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=False,
        )) as sess:
            train_data = DataManage.DataManage(train_frames, train_targets, batch_size)
            model = Model(n_gpu=n_gpu, model_name='ctdnn', n_speaker=train_data.spkr_num)
            initial = tf.global_variables_initializer()
            sess.run(initial)
            saver = tf.train.Saver()
            for i in range(max_step):
                _, loss = sess.run(model.train_step(train_data, lr))
                print(i, " loss:", loss)
                if i % 25 == 0 or i + 1 == max_step:
                    saver.save(sess, save_path)
