import tensorflow as tf
import sys
sys.path.append("../..")
import os
import time
import numpy as np
from pyasv.data_manage import DataManage
from pyasv.data_manage import DataManage4BigData


class MaxFeatureMapDnn:
    def __init__(self, config, x, y=None):
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
        if y is not None:
            self._build_train_graph(x, y)
        else:
            self._build_pred_graph(x)


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

    def _build_pred_graph(self, x):
        _, self._feature = self._inference(x)

    def _build_train_graph(self, x, y):
        out, mfm6 = self._inference(x)
        self._feature = out
        self._prediction = out
        self._loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out)

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
                                strides=[1, 1, 1, 1], padding='SAME')
        mfm_4_b = self._max_feature_map(conv_4_b)
        #

        conv_5_a = self._conv2d(mfm_4_b, 'Conv5_a', shape=[1, 1, 64, 128], 
                                strides=[1, 1, 1, 1], padding='SAME')
        mfm_5_a = self._max_feature_map(conv_5_a)
        conv_5_b = self._conv2d(mfm_5_a, 'Conv5_b', shape=[3, 3, 64, 128], 
                                strides=[1, 1, 1, 1], padding='SAME')
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
            weights = self._weights_variable(shape, name=name+'w')
        return tf.nn.conv1d(x, weights, stride=strides, padding='SAME', name=name+"_output")

    def _conv2d(self, x, name, shape, strides, padding='SAME'):
        with tf.name_scope(name):
            weights = self._weights_variable(shape, name=name+'w')
            biases = self._bias_variable(shape[-1], name=name+'b')
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights,
                                                      strides=strides, padding=padding), biases, name=name+"_output"))

    def _full_connect(self, x, name, units):
        with tf.name_scope(name):
            weights = self._weights_variable([x.get_shape().as_list()[-1], units], name=name+'w')
            biases = self._bias_variable(units, name=name+'b')
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), biases), name=name+"_output")

    @staticmethod
    def _weights_variable(shape, name='weights', stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.get_variable(initializer=initial, name=name)

    @staticmethod
    def _bias_variable(shape, name='bias', stddev=0.1):
        initial = tf.truncated_normal([shape], stddev=stddev)
        return tf.get_variable(initializer=initial, name=name)

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


def cosine(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))


def average_losses(loss):
    tf.add_to_collection('losses', loss)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses')

    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y):
    for i in range(len(models)):
        x, y, _, _, _, _ = models[i]
        start_pos = i * payload_per_gpu
        stop_pos = (i + 1) * payload_per_gpu
        inp_dict[x] = batch_x[start_pos:stop_pos]
        inp_dict[y] = batch_y[start_pos:stop_pos]
    return inp_dict


def _no_gpu(config, train, validation):
    tf.reset_default_graph()
    with tf.Session() as sess:
        learning_rate = config.LR
        print('build model...')
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        x = tf.placeholder(tf.float32, [None, 50, 40, 1])
        y = tf.placeholder(tf.float32, [None, config.N_SPEAKER])
        model = MaxFeatureMapDnn(config, x, y)
        pred = model.prediction
        loss = model.loss
        feature = model.feature
        train_op = opt.minimize(loss)
        vectors = dict()
        print("done...")
        print('run train op...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(config.MAX_STEP):
            start_time = time.time()
            avg_loss = 0.0
            total_batch = int(train.num_examples / config.BATCH_SIZE) - 1
            print('\n---------------------')
            print('Epoch:%d, lr:%.4f' % (epoch, config.LR))
            feature_ = None
            ys = None
            for batch_id in range(total_batch):
                batch_x, batch_y = train.next_batch
                batch_x = batch_x.reshape(-1, 50, 40, 1)
                batch_y = np.eye(train.spkr_num)[batch_y.reshape(-1)]
                _, _loss, feature = sess.run([train_op, loss, feature],
                                             feed_dict={x: batch_x, y: batch_y})
                avg_loss += _loss
                if ys is None:
                    ys = batch_y
                else:
                    ys = np.concatenate((ys, batch_y), 0)
                if feature_ is None:
                    feature_ = feature
                else:
                    feature_ = np.concatenate((feature_, feature), 0)
                print("batch_%d  batch_loss=%.4f"%(batch_id, _loss), end='\r')
            print('\n')
            train.reset_batch_counter()
            for spkr in range(config.N_SPEAKER):
                if len(feature_[np.argmax(ys, 1) == spkr]):
                    vector = np.mean(feature_[np.argmax(ys, 1) == spkr], axis=0)
                    if spkr in vectors.keys():
                        vector = (vectors[spkr] + vector) / 2
                    else:
                        vector = vector
                    vectors[spkr] = vector
                else:
                    if spkr not in vectors.keys():
                        vectors[spkr] = np.zeros(400, dtype=np.float32)
            avg_loss /= total_batch
            print('Train loss:%.4f' % (avg_loss))
            total_batch = int(validation.num_examples / config.BATCH_SIZE) - 1
            preds = None
            feature = None
            ys = None
            for batch_idx in range(total_batch):
                print("validation in batch_%d..." % batch_idx, end='\r')
                batch_x, batch_y = validation.next_batch
                batch_x = batch_x.reshape(-1, 50, 40, 1)
                batch_y, batch_pred, batch_feature = sess.run([y, pred, feature],
                                                              feed_dict={x: batch_x, y: batch_y})
                if preds is None:
                    preds = batch_pred
                else:
                    preds = np.concatenate((preds, batch_pred), 0)
                if feature is None:
                    feature = batch_feature
                else:
                    feature = np.concatenate((feature, batch_feature), 0)
                if ys is None:
                    ys = batch_y
                else:
                    ys = np.concatenate((ys, batch_y), 0)
            vec_preds = []
            validation.reset_batch_counter()
            for sample in range(feature.shape[0]):
                score = -100
                pred = -1
                for spkr in vectors.keys():
                    if cosine(vectors[spkr], feature[sample]) > score:
                        score = cosine(vectors[spkr], feature[sample])
                        pred = int(spkr)
                vec_preds.append(pred)
            correct_pred = np.equal(np.argmax(ys, 1), vec_preds)
            val_accuracy = np.mean(np.array(correct_pred, dtype='float'))
            print('Val Accuracy: %0.4f%%' % (100.0 * val_accuracy))
            saver.save(sess=sess, save_path=os.path.join(model._save_path, model._name))
            stop_time = time.time()
            elapsed_time = stop_time - start_time
            print('Cost time: ' + str(elapsed_time) + ' sec.')
        print('training done.')


def _multi_gpu(config, train, validation):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            learning_rate = config.LR
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            print('build model...')
            print('build model on gpu tower...')
            models = []
            for gpu_id in range(config.N_GPU):
                with tf.device('/gpu:%d' % gpu_id):
                    print('tower:%d...' % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('cpu_variables', reuse=tf.AUTO_REUSE):
                            x = tf.placeholder(tf.float32, [None, 50, 40, 1])
                            y = tf.placeholder(tf.float32, [None, train.spkr_num])
                            model = MaxFeatureMapDnn(config, x, y)
                            pred = model.prediction
                            feature = model.feature
                            loss = model.loss
                            grads = opt.compute_gradients(loss)
                            models.append((x, y, pred, loss, grads, feature))
            print('build model on gpu tower done.')

            print('reduce model on cpu...')
            tower_x, tower_y, tower_preds, tower_losses, tower_grads, tower_feature = zip(*models)
            aver_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))
            get_feature = tf.reshape(tf.stack(tower_feature, 0), [-1, 400])

            all_y = tf.reshape(tf.stack(tower_y, 0), [-1, config.N_SPEAKER])

            all_pred = tf.reshape(tf.stack(tower_preds, 0), [-1, config.N_SPEAKER])

            vectors = dict()

            print('reduce model on cpu done.')

            print('run train op...')
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

            for epoch in range(config.MAX_STEP):
                start_time = time.time()
                payload_per_gpu = int(config.BATCH_SIZE // config.N_GPU)
                if config.BATCH_SIZE % config.N_GPU:
                    print("Warning: Batch size can't to be divisible of N_GPU")
                total_batch = int(train.num_examples / config.BATCH_SIZE) - 1
                avg_loss = 0.0
                print('\n---------------------')
                print('Epoch:%d, lr:%.4f' % (epoch, config.LR))
                feature_ = None
                ys = None
                for batch_idx in range(total_batch):
                    batch_x, batch_y = train.next_batch
                    batch_x = batch_x.reshape(-1, 50, 40, 1)
                    inp_dict = dict()
                    # print("data part done...")
                    inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y)
                    _, _loss, feature = sess.run([apply_gradient_op, aver_loss_op, get_feature], inp_dict)
                    # print("train part done...")
                    avg_loss += _loss
                    if ys is None:
                        ys = batch_y
                    else:
                        ys = np.concatenate((ys, batch_y), 0)
                    if feature_ is None:
                        feature_ = feature
                    else:
                        feature_ = np.concatenate((feature_, feature), 0)
                    print("batch_%d  batch_loss=%.4f"%(batch_idx, _loss), end='\r')
                print('\n')
                train.reset_batch_counter()
                for spkr in range(config.N_SPEAKER):
                    if len(feature_[np.argmax(ys, 1) == spkr]):
                        vector = np.mean(feature_[np.argmax(ys, 1) == spkr], axis=0)
                        if spkr in vectors.keys():
                            vector = (vectors[spkr] + vector) / 2
                        else:
                            vector = vector
                        vectors[spkr] = vector
                    else:
                        if spkr not in vectors.keys():
                            vectors[spkr] = np.zeros(400, dtype=np.float32)
                    # print("vector part done....")
                avg_loss /= total_batch
                print('Train loss:%.4f' % (avg_loss))

                val_payload_per_gpu = int(config.BATCH_SIZE // config.N_GPU)
                if config.BATCH_SIZE % config.N_GPU:
                    print("Warning: Batch size can't to be divisible of N_GPU")

                total_batch = int(validation.num_examples / config.BATCH_SIZE) - 1
                preds = None
                ys = None
                feature = None
                for batch_idx in range(total_batch):

                    batch_x, batch_y = validation.next_batch
                    batch_x = batch_x.reshape(-1, 50, 40, 1)
                    inp_dict = feed_all_gpu({}, models, val_payload_per_gpu, batch_x, batch_y)
                    batch_pred, batch_y, batch_feature = sess.run([all_pred, all_y, get_feature], inp_dict)
                    if preds is None:
                        preds = batch_pred
                    else:
                        preds = np.concatenate((preds, batch_pred), 0)
                    if feature is None:
                        feature = batch_feature
                    else:
                        feature = np.concatenate((feature, batch_feature), 0)
                    if ys is None:
                        ys = batch_y
                    else:
                        ys = np.concatenate((ys, batch_y), 0)

                vec_preds = []
                validation.reset_batch_counter()
                for sample in range(feature.shape[0]):
                    score = -100
                    pred = -1
                    for spkr in vectors.keys():
                        if cosine(vectors[spkr], feature[sample]) > score:
                            score = cosine(vectors[spkr], feature[sample])
                            pred = int(spkr)
                    vec_preds.append(pred)
                correct_pred = np.equal(np.argmax(ys, 1), vec_preds)
                val_accuracy = np.mean(np.array(correct_pred, dtype='float'))
                print('Val Accuracy: %0.4f%%' % (100.0 * val_accuracy))
                saver.save(sess=sess, save_path=os.path.join(model._save_path, model._name))
                stop_time = time.time()
                elapsed_time = stop_time - start_time
                print('Cost time: ' + str(elapsed_time) + ' sec.')
            print('training done.')


def run(config, train, validation):
    """Train MFM model.
    
    Parameters
    ----------
    config : ``config``
        the config of model.
    train : ``DataManage``
        train dataset.
    validation
        validation dataset.
    """
    if config.N_GPU == 0:
        _no_gpu(config, train, validation)
    else:
        if os.path.exists('./tmp'):
            os.rename('./tmp', './tmp-backup')
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
        memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_list = []
        for gpu in range(config.N_GPU):
            gpu_list.append(str(np.argmax(memory_gpu)))
            memory_gpu[np.argmax(memory_gpu)] = -10000
        s = ""
        for i in range(config.N_GPU):
            if i != 0:
                s += ','
            s += str(gpu_list[i])
        os.environ['CUDA_VISIBLE_DEVICES'] = s
        os.remove('./tmp')
        _multi_gpu(config, train, validation)


def restore():
    print("not implemented now")


def _main():
    """
    Test model.
    """
    from pyasv.data_manage import DataManage
    from pyasv import Config
    sys.path.append("../..")

    con = Config(name='ctdnn', n_speaker=100, batch_size=64, n_gpu=4, max_step=20, is_big_dataset=False,
                 learning_rate=0.001, save_path='./save')
    x = np.random.random([6400, 50, 40, 1])
    y = np.random.randint(0, 100, [6400, 1])
    train = DataManage(x, y, con)

    x = np.random.random([640, 50, 40, 1])
    y = np.random.randint(0, 100, [640, 1])
    validation = DataManage(x, y, con)

    run(con, train, validation)


if __name__ == '__main__':
    _main()