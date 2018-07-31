import os
import sys
sys.path.append("../..")
import numpy as np
import tensorflow as tf
import pyasv.loss.triplet_loss as triplet_loss
import time
from pyasv.data_manage import DataManage
from pyasv.data_manage import DataManage4BigData
from tensorflow.python import debug


class DeepSpeaker:
    def __init__(self, config, x, y=None, out_channel=[64, 128, 256, 512]):
        """Create deep speaker model.

        Parameters
        ----------
        config : ``config`` class.
            The config of ctdnn model, no extra need.
        out_channel : ``list``
            The out channel of each res_block.::
            
                out_channel = [64, 128, 256, 512]

        Notes
        -----
        There is no predict operation in deep speaker model.
        Because we use the output of last layer as speaker vector.
        we can use DeepSpeaker.feature to get these vector.
        """
        self._gpu_ind = 0
        self.n_blocks = len(out_channel)
        self.out_channel = out_channel
        self._name = config.MODEL_NAME
        self._n_speaker = config.N_SPEAKER
        self._max_step = config.MAX_STEP
        self._n_gpu = config.N_GPU
        self._conv_weight_decay = config.CONV_WEIGHT_DECAY
        self._fc_weight_decay = config.FC_WEIGHT_DECAY
        self._save_path = config.SAVE_PATH
        self._bn_epsilon = config.BN_EPSILON
        self._learning_rate = config.LR
        self._batch_size = config.BATCH_SIZE
        self._vectors = dict()
        if y is not None:
            self._build_train_graph(x, y)
        else:
            self._build_pred_graph(x)

    @property
    def feature(self):
        """A ``property`` member to get feature.

        Returns
        -------
        feature : ``tf.operation``
        """
        return self._feature

    @property
    def loss(self):
        """A ``property`` member to get loss.

        Returns
        -------
        loss : ``tf.operation``
        """
        return self._loss

    def _build_pred_graph(self, x):
        output = self._inference(x)
        self._feature = output

    def _build_train_graph(self, x, y):
        output = self._inference(x)
        self._feature = output
        self._loss = self._triplet_loss(output, y)
        
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

    def _residual_block(self, inp, out_channel, name, is_first_layer=False):
        print(name)
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
        #loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=targets,
        #                                                               embeddings=inp)
        return loss

    def _batch_normalization(self, inp, name):
        bn_layer = tf.layers.batch_normalization(inp, epsilon=self._bn_epsilon)
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
        x, y, _, _, _ = models[i]
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
        x = tf.placeholder(tf.float32, [None, 100, 64, 1])
        y = tf.placeholder(tf.float32, [None, config.N_SPEAKER])
        model = DeepSpeaker(config=config, x=x, y=y)
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
            total_batch = int(train.num_examples / config.BATCH_SIZE)
            print('\n---------------------')
            print('Epoch:%d, lr:%.4f' % (epoch, config.LR))
            feature_ = None
            for batch_id in range(total_batch):
                batch_x, batch_y = train.next_batch
                batch_x = batch_x.reshape(-1, 9, 40, 1)
                batch_y = np.eye(train.spkr_num)[batch_y.reshape(-1)]
                _, _loss, feature = sess.run([train_op, loss, feature],
                                             feed_dict={x: batch_x, y: batch_y})
                avg_loss += _loss
                if feature_ is None:
                    feature_ = feature
                else:
                    feature_ = np.concatenate((feature_, feature), 0)
                print("batch_%d  batch_loss=%.4f"%(batch_id, _loss), end='\r')
            for spkr in range(config.N_SPEAKER):
                if len(feature[np.argmax(batch_y, 1) == spkr]):
                    vector = np.mean(feature[np.argmax(batch_y, 1) == spkr], axis=0)
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
            total_batch = int(validation.num_examples / config.N_GPU)
            ys = None
            feature = None
            for batch_idx in range(total_batch):
                print("validation in batch_%d..."%batch_idx, end='\r')
                batch_x, batch_y = validation.next_batch
                batch_y, batch_feature = sess.run([y, feature],
                                                  feed_dict={x: batch_x, y: batch_y})
                if feature is None:
                    feature = batch_feature
                else:
                    feature = np.concatenate((feature, batch_feature), 0)
                if ys is None:
                    ys = batch_y
                else:
                    ys = np.concatenate((ys, batch_y), 0)
            vec_preds = []
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
            stop_time = time.time()
            elapsed_time = stop_time-start_time
            print('Cost time: ' + str(elapsed_time) + ' sec.')
            saver.save(sess=sess, save_path=os.path.join(model._save_path, model._name))
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
                            x = tf.placeholder(tf.float32, [None, 100, 64, 1])
                            y = tf.placeholder(tf.float32, [None, config.N_SPEAKER])
                            model = DeepSpeaker(config, x, y)
                            feature = model.feature
                            loss = model.loss
                            grads = opt.compute_gradients(loss)
                            models.append((x, y, loss, grads, feature))
            print('build model on gpu tower done.')

            print('reduce model on cpu...')
            tower_x, tower_y, tower_losses, tower_grads, tower_feature = zip(*models)
            aver_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))
            get_feature = tf.reshape(tf.stack(tower_feature, 0), [-1, 400])

            all_y = tf.reshape(tf.stack(tower_y, 0), [-1, config.N_SPEAKER])

            vectors = dict()

            print('reduce model on cpu done.')

            print('run train op...')
            sess.run(tf.global_variables_initializer())

            # debug_mode
            # sess = debug.LocalCLIDebugWrapperSession(sess=sess)

            saver = tf.train.Saver()
            
            for epoch in range(config.MAX_STEP):
                start_time = time.time()
                payload_per_gpu = int(config.BATCH_SIZE//config.N_GPU)
                if config.BATCH_SIZE % config.N_GPU:
                    print("Warning: Batch size can't to be divisible of N_GPU")
                total_batch = int(train.num_examples / config.BATCH_SIZE)
                avg_loss = 0.0
                print('\n---------------------')
                print('Epoch:%d, lr:%.4f' % (epoch, config.LR))
                feature_ = None
                for batch_idx in range(total_batch):
                    batch_x, batch_y = train.next_batch
                    batch_x = batch_x.reshape(-1, 9, 40, 1)
                    inp_dict = dict()
                    # print("data part done...")
                    inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y)
                    _, _loss, feature = sess.run([apply_gradient_op, aver_loss_op, get_feature], inp_dict)
                    # print("train part done...")
                    avg_loss += _loss
                    if feature_ is None:
                        feature_ = feature
                    else:
                        feature_ = np.concatenate((feature_, feature), 0)
                    print("batch_%d  batch_loss=%.4f"%(batch_idx, _loss), end='\r')
                for spkr in range(config.N_SPEAKER):
                    if len(feature[np.argmax(batch_y, 1) == spkr]):
                        vector = np.mean(feature[np.argmax(batch_y, 1) == spkr], axis=0)
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

                val_payload_per_gpu = int(config.BATCH_SIZE//config.N_GPU)
                if config.BATCH_SIZE % config.N_GPU:
                    print("Warning: Batch size can't to be divisible of N_GPU")
                
                total_batch = int(validation.num_examples / config.N_GPU)
                ys = None
                feature = None
                for batch_idx in range(total_batch):

                    batch_x, batch_y = validation.next_batch

                    inp_dict = feed_all_gpu({}, models, val_payload_per_gpu, batch_x, batch_y)
                    batch_y, batch_feature = sess.run([all_y, get_feature], inp_dict)
                    if feature is None:
                        feature = batch_feature
                    else:
                        feature = np.concatenate((feature, batch_feature), 0)
                    if ys is None:
                        ys = batch_y
                    else:
                        ys = np.concatenate((ys, batch_y), 0)

                vec_preds = []
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
                elapsed_time = stop_time-start_time
                print('Cost time: ' + str(elapsed_time) + ' sec.')
            print('training done.')


def cosine(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))


def run(config, train, validation):
    """Train DeepSpeaker model.
    
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
        memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
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
    import sys
    sys.path.append("../..")
    con = Config(name='deepspeaker', n_speaker=100, batch_size=64, n_gpu=4, max_step=20, is_big_dataset=False,
                 learning_rate=0.001, save_path='./save', conv_weight_decay=0.01, fc_weight_decay=0.01, bn_epsilon=1e-3)
    x = np.random.random([6400, 100, 64, 1])
    y = np.random.randint(0, 100, [6400, 1])
    train = DataManage(x, y, con)

    x = np.random.random([640, 100, 64, 1])
    y = np.random.randint(0, 100, [640, 1])
    validation = DataManage(x, y, con)

    run(con, train, validation)


if __name__ == '__main__':
    _main()
