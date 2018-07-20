import tensorflow as tf
import sys
sys.path.append("..")
from scipy.spatial.distance import cosine
import os
import time
import numpy as np
from pyasv.data_manage import DataManage
from pyasv.data_manage import DataManage4BigData

class DPN:
    def __init__(self, config, cardinality, filter_increase, initial_conv_filters, depth, width):
        """
        depth: list, contain the bn comp of each block
        width: the width of filter
        cardinality: base number
        config: config type
        filter_increase: list, contain the increase dims of each block
        """
        self._width = width
        self._cardinality = cardinality
        self._filter_increase = filter_increase
        self._depth = depth
        self._initial_conv_filters = initial_conv_filters
        self._config = config
        self._batch_size = config.BATCH_SIZE
        self._n_gpu = config.N_GPU
        self._name = config.MODEL_NAME
        self._n_speaker = config.N_SPEAKER
        self._max_step = config.MAX_STEP
        self._is_big_dataset = config.IS_BIG_DATASET
        self._weight_decay = config.WEIGHT_DECAY
        if self._is_big_dataset:
            self._url_of_big_dataset = config.URL_OF_BIG_DATASET
        self._lr = config.LR
        self._save_path = config.SAVE_PATH
        self._bn_epsilon = config.BN_EPSILON
        self._n_blocks = len(list(depth))
        
    @property
    def loss(self):
        return self._loss

    @property
    def prediction(self):
        return self._prediction

    @property
    def feature(self):
        return self._feature

    def _build_pred_graph(self):
        pred_frames = tf.placeholder(tf.float32, shape=[None, 9, 40, 1], name='pred_x')
        frames = pred_frames
        _, self._feature = self._inference(frames)   

    def _build_train_graph(self):
        """
        Build the compute graph.
        """
        self._gpu_ind = tf.get_variable(name='gpu_ind', trainable=False 
                            ,shape=[],dtype=tf.int32, initializer=tf.constant_initializer(value=0, dtype=tf.int32))
        batch_frames = tf.placeholder(tf.float32, shape=[self._n_gpu, None, 9, 40, 1],name='x')
        batch_target = tf.placeholder(tf.float32, shape=[self._n_gpu, None, self._n_speaker],name='y_')
        frames = batch_frames[self._gpu_ind]
        target = batch_target[self._gpu_ind]
        self._prediction, self._feature = self._inference(frames)
        self._loss = -tf.reduce_mean(target * tf.log(tf.clip_by_value( self._prediction, 1e-8, tf.reduce_max(self._prediction))))

        self._opt = tf.train.AdamOptimizer(self._lr)
        self._opt.minimize(self._loss)
        self._saver = tf.train.Saver()
        
    def _inference(self, frames):
        base_filter = 256
        filter_inc = self._filter_increase[0]
        filters = int(self._cardinality * self._width)
        N = self._depth
        x = self._initial_conv_block(frames, self._initial_conv_filters, self._weight_decay)

        x = self._dual_path_block(x, filters, filters, base_filter, filter_inc, self._cardinality, 'projection')

        for i in range(N[0] - 1):
            x = self._dual_path_block(x, filters, filters, base_filter, filter_inc, self._cardinality)
        for k in range(1, len(N)):
            print("BLOCK %d" % (k + 1))
            filter_inc = self._filter_increase[k]
            filters *= 2
            base_filter *= 2
            x = self._dual_path_block(x, pointwise_filters_a=filters,
                                      grouped_conv_filters_b=filters,
                                      pointwise_filters_c=base_filter,
                                      filter_increase=filter_inc,
                                      cardinality=self._cardinality,
                                      block_type='downsample')

            for i in range(N[k] - 1):
                x = self._dual_path_block(x, pointwise_filters_a=filters,
                                          grouped_conv_filters_b=filters,
                                          pointwise_filters_c=base_filter,
                                          filter_increase=filter_inc,
                                          cardinality=self._cardinality,
                                          block_type='normal')
        x = tf.concat(x)

        x = tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 1, 2, 1], padding='VALID')
            
        x = tf.reshape(x, [-1,
                           x.get_shape().as_list()[1] *
                           x.get_shape().as_list()[2] *
                           x.get_shape().as_list()[3]])
        x = self._full_connect(x, 'feature_layer', 400)
        feature = x
        output = tf.nn.softmax(x)
        return output, feature
        
    def _initial_conv_block(self, x,
                            initial_conv_filters,
                            weight_decay):
        x = self._conv2d(x, 'init_conv', initial_conv_filters, strides=[1, 7, 7, 1], padding='SAME')

        x = self._batch_normalization(x, 'initial_bn')
        # x = BatchNormalization(axis=channel_axis)(x)
        # x = Activation('relu')(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])
        #x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        return x

    def _dual_path_block(self, x, 
                         pointwise_filters_a, 
                         grouped_conv_filters_b, 
                         pointwise_filters_c,
                         filter_increase,
                         cardinality,
                         block_type='normal'):
        grouped_channels = int(grouped_conv_filters_b / cardinality)
        if isinstance(x, list):
            init = tf.concat(x, axis=3) 
        else:
            init = x

        if block_type == 'projection':
            stride = [1, 1, 1, 1]
            projection = True
        elif block_type == 'downsample':
            stride = [1, 2, 2, 1]
            projection = True
        elif block_type == 'normal':
            stride = [1, 1, 1, 1]
            projection = False
        else:
            raise ValueError('`block_type` must be one of ["projection", "downsample", "normal"]. '
                             'but you give %s' % block_type)

        if projection:
            projection_path = self._bn_relu_conv_block(init, name='project_block',
                                                       filter_shape=[1, 1, init.get_shape().as_list()[3],
                                                                     pointwise_filters_c + 2 * filter_increase],
                                                       padding='SAME', stride=stride)
            input_residual_path = tf.keras.layers.Lambda(lambda z: z[:, :, :, :pointwise_filters_c])(projection_path)
            input_dense_path = tf.keras.layers.Lambda(lambda z: z[:, :, :, pointwise_filters_c:])(projection_path)
        else:
            input_residual_path = x[0]
            input_dense_path = x[1]

        x = self._bn_relu_conv_block(x, name='normal_block_first_comp',
                                     filter_shape=[1, 1, init.get_shape().as_list()[3], pointwise_filters_a], 
                                     padding='SAME',
                                     stride=stride)
        x = self._grouped_convolution_block(x, grouped_channels=grouped_channels,
                                            cardinality=cardinality, strides=stride)
        x = self._bn_relu_conv_block(x, name='normal_block_second_comp',
                                     filter_shape=[1, 1, x.get_shape().as_list()[3],
                                                   pointwise_filters_c + 2 * filter_increase],
                                     padding='SAME',
                                     stride=stride)

        output_residual_path = tf.keras.layers.Lambda(lambda z: z[:, :, :, :pointwise_filters_c])(x)
        output_dense_path = tf.keras.layers.Lambda(lambda z: z[:, :, :, pointwise_filters_c:])(x)

        residual_path = tf.add(input_residual_path, output_residual_path)
        dense_path = tf.concat([input_dense_path, output_dense_path], axis=3)

        return [residual_path, dense_path]

    def _grouped_convolution_block(self, x, grouped_channels, cardinality, strides):
        group_list = []
        if cardinality == 1:
            # with cardinality 1, it is a standard convolution
            x = self._conv2d(x, name='group_conv', shape=[3, 3, grouped_channels, grouped_channels], strides=[1,1,1,1], padding='same')
            x = self._batch_normalization(x, 'group_merge')
            x = tf.nn.relu(x)
            return x

        for c in range(cardinality):
            x = tf.keras.layers.Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(x)
            x = self._conv2d(x, name='group_conv', shape=[3, 3, grouped_channels, grouped_channels], strides=[1,1,1,1],padding='same')
            group_list.append(x)

        group_merge = tf.concat(group_list, axis=3)
        group_merge = self._batch_normalization(group_merge, 'group_merge')
        group_merge = tf.nn.relu(group_merge)
        return group_merge

    def _bn_relu_conv_block(self, inp, filter_shape, stride, padding, name):
        weight = self._weights_variable(name+"_filter", filter_shape, "Conv")
        bn_layer = self._batch_normalization(inp, name)
        relu_layer = tf.nn.relu(bn_layer)
        conv_layer = tf.nn.conv2d(relu_layer, weight,
                                  strides=stride, padding=padding)
        return conv_layer

    def _batch_normalization(self, inp, name):
        dims = inp.get_shape()[-1]
        mean, variance = tf.nn.moments(inp, axes=[0, 1, 2])
        beta = tf.get_variable(name+'_beta', dims, tf.float32,
                               initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable(name+'_gamma', dims, tf.float32,
                                initializer=tf.constant_initializer(value=0.0))
        bn_layer = tf.nn.batch_normalization(inp, mean, variance, beta, gamma, self._bn_epsilon)
        return bn_layer

    def _conv2d(self, x, name, shape, strides, padding='SAME'):
        with tf.name_scope(name):
            weights = self._weights_variable(shape, name+'_w')
            biases = self._bias_variable(shape[-1])
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights,
                                                      strides=strides, padding=padding), biases, name=name+"_output"))

    def _full_connect(self, x, name, units):
        with tf.name_scope(name):
            weights = self._weights_variable([x.get_shape().as_list()[-1], units], name)
            biases = self._bias_variable(units)
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), biases), name=name+"_output")

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

    def _weights_variable(self, shape, name, initial="var"):
        if initial == "var":
            return tf.get_variable(name=name, shape=shape,
                                   initializer=tf.variance_scaling_initializer())
        else:
            return tf.get_variable(name=name, shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer())
    
    def _bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

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
        enroll_f = [enroll_frames]
        for i in range(self._n_gpu-1):
            enroll_f.append(np.zeros(np.array(enroll_frames).shape))
        enroll_l = [enroll_labels]
        for i in range(self._n_gpu-1):
            enroll_l.append(np.zeros(np.array(enroll_labels).shape))
        
        features = sess.run(self.feature, feed_dict={'x:0':np.array(enroll_f).reshape([4, None, 9, 40, 1]), 'y:0':enroll_l})
        print(features)
        return features

    def run(self,
            train_frames, 
            train_targets,
            enroll_frames=None,
            enroll_labels=None,
            test_frames=None,
            test_labels=None,
            need_prediction_now=False):
        
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=False,
            )) as sess:
                # initial tensorboard
                writer = tf.summary.FileWriter(os.path.join(self._save_path, 'graph'),sess.graph)
    
                # prepare data
                if self._is_big_dataset:
                    train_data = DataManage4BigData(self._url_of_big_dataset, self._config)
                    if not train_data.file_is_exist:
                        train_data.write_file(train_frames, train_targets)
                    del train_frames, train_targets
                    self._build_train_graph()
                else:
                    self._build_train_graph()
                    train_data = DataManage(train_frames, train_targets, self._batch_size-1)
                
                # initial step
                initial = tf.global_variables_initializer()
                sess.run(initial)
                train_op, loss = self._train_step()
                if enroll_frames:
                    accuracy = self._validation_acc(sess, enroll_frames, enroll_labels, test_frames, test_labels)
                
                # record the memory usage and time of each step
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                # define tensorboard steps
                loss_summary = tf.summary.scalar('loss_summary', loss)
                if enroll_frames:
                    acc_summary = tf.summary.scalar('accuracy', accuracy)
                merged_summary = tf.summary.merge_all()

                last_time = time.time()
                
                # train loop
                for i in range(self._max_step):
                    # get data
                    input_frames = []
                    input_labels = []
                    for x in range(self._n_gpu):
                        frames, labels = train_data.next_batch
                        input_frames.append(frames)
                        input_labels.append(labels)
                    input_frames = np.array(input_frames).reshape([self._n_gpu, self._batch_size, 9, 40, 1])
                    input_labels = np.array(input_labels).reshape([self._n_gpu, self._batch_size, self._n_speaker])
                    
                    _, summary_str = sess.run([train_op, merged_summary], feed_dict={'x:0':input_frames, 'y_:0':input_labels})
                    current_time = time.time()
                    
                    # print log
                    print("-------")
                    print("No.%d step use %f sec"%(i,current_time-last_time))
                    print("loss = %f" % loss)
                    last_time = time.time()
                    
                    # record
                    if i % 10 == 0 or i + 1 ==self._max_step:
                        self._saver.save(sess, os.path.join(self._save_path,'model'))
                    
                    writer.add_run_metadata(run_metadata,'step%d'%i)
                    writer.add_summary(summary_str, i)
        
        if need_prediction_now:
            self.run_predict(self._save_path, enroll_frames, enroll_labels, test_frames, test_labels)
        writer.close()

    def run_predict(self, 
                    save_path,
                    enroll_frames,
                    enroll_targets, 
                    test_frames,
                    test_label):
        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                self._build_pred_graph()
                new_saver = tf.train.Saver()

                # needn't batch and gpu in prediction
                
                enroll_data = DataManage(enroll_frames, enroll_targets, self._batch_size)
                test_data = DataManage(test_frames, test_label, self._batch_size)
                new_saver.restore(sess, tf.train.latest_checkpoint(self._save_path))
                feature_op = graph.get_operation_by_name('feature_layer_output')
                vector_dict = dict()
                while not enroll_data.is_eof:
                    frames, labels = enroll_data.next_batch
                    frames = np.array(frames).reshape([-1, 9, 40, 1])
                    labels = np.array(labels).reshape([-1, self._n_speaker])
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
                    labels = np.array(labels).reshape([-1, self._n_speaker])
                    vectors = sess.run(feature_op, feed_dict={'pred_x:0': frames})
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
