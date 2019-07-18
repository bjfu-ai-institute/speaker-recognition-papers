import logging
import os
import tensorflow as tf
import numpy as np
import multiprocessing as mp
from tensorflow.contrib import slim


class TFrecordGen:
    def __init__(self, config, file_name, lazy_open=False):
        self.log = logging.getLogger('data')
        self.url = config.save_path
        self.num_threads = config.n_threads
        self.file_name = file_name
        self.lazy_open = lazy_open
        if not lazy_open:
            self.writer = tf.python_io.TFRecordWriter(os.path.join(self.url, 'data', self.file_name))

    def write(self, data, label, data_tpye='float', label_type='int'):
        label = np.array(label, dtype=np.int32)
        if self.lazy_open:
            self.writer = tf.python_io.TFRecordWriter(os.path.join(self.url, 'data', self.file_name))
        output_file = os.path.join(self.url, 'data', self.file_name)
        self.log.info("Writing to %s, received data shape is %s, label shape is %s"%(output_file,
                                                                                  np.array(data).shape[0],
                                                                                  np.array(label).shape[0]))
        for i, j in zip(data, label):
            feature_dict = {}
            #feature_dict = {
            #    'data': tf.train.Feature(float_list=tf.train.FloatList(value=np.array(i).reshape(-1,))),
            #    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(j).reshape(-1,)))
            #}
            if data_tpye == 'int':
                feature_dict['data'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(i).reshape(-1,)))
            elif data_tpye == 'float':
                feature_dict['data'] = tf.train.Feature(float_list=tf.train.FloatList(value=np.array(i).reshape(-1,)))
            else:
                raise TypeError("Data type should be one of ['float', 'int']")
            if label_type == 'int':
                feature_dict['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(j).reshape(-1,)))
            elif label_type == 'float':
                feature_dict['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=np.array(j).reshape(-1,)))
            else:
                raise TypeError("Data type should be one of ['float', 'int']")
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            self.writer.write(example.SerializeToString())
        self.log.info("Writing to %s finished." % output_file)
        if self.lazy_open:
            self.close()

    def close(self):
        self.writer.close()


class TFrecordReader:
    def __init__(self, files, data_shape, label_shape, descriptions=None, raw=False):
        self.no_label = (label_shape is None)
        self.data_files = files
        if label_shape is not None:
            self.keys_to_features = {'data': tf.FixedLenFeature(shape=data_shape, dtype=tf.float32)}
        else:
            self.keys_to_features = {'data': tf.FixedLenFeature(shape=data_shape, dtype=tf.float32),
                                     'label': tf.FixedLenFeature(shape=label_shape, dtype=tf.int64)}

    def parse(self, proto):
        parsed_ = tf.parse_single_example(proto, self.keys_to_features)
        if self.no_label:
            return parsed_['data']
        return parsed_['data'], parsed_['label']

    def read(self, batch_size, repeat=True, shuffle=False):
        dataset = tf.data.TFRecordDataset(self.data_files)
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.map(self.parse)
        dataset = dataset.batch(batch_size)
        return dataset

    def read_raw(self):
        dataset = tf.data.TFRecordDataset(self.data_files)
        return dataset


class TFrecordClassBalanceGen:
    def __init__(self, config, file_name):
        self.class_num = config.n_speaker
        self.log = logging.getLogger('data')
        self.n_threads = config.n_threads
        self.writer = [LazyWriter(config, file_name+'%d.rcd'%i) for i in range(self.class_num)]

    def write(self, data, label):
        data = np.array(data)
        label = np.array(label, dtype=np.int32)
        if len(label.shape) == 2 and label.shape[-1] == 1:
            label = label.reshape(-1,)
        else:
            raise ValueError("label's shape must be 1-d array or 2-d array which like (x, 1)")
        data = [data[np.where(label == i)[0]] for i in range(self.class_num)]
        label = [np.full(shape=data[i].shape[0], fill_value=i) for i in range(self.class_num)]
        print(data)
        for i in range(len(data)):
            if data[i].shape[0] != 0:
                self.writer[i].write(data[i], label[i])


class LazyWriter(TFrecordGen):
    def __init__(self, config, file_name):
        super().__init__(config, file_name, True)



class TFrecordClassBalanceReader:
    def __init__(self, config, filenames, data_shape=[]):
        self.data_shape = data_shape
        self.feature_dim = config.feature_dims
        datasets = [tf.data.TFRecordDataset(f).map(self.parse, num_parallel_calls=16).repeat(None) for f in filenames]
        num_classes_per_batch = config.num_classes_per_batch
        num_utt_per_class = config.num_utt_per_class
        num_classes = config.n_speaker
        def generator(_):
            # Sample `num_classes_per_batch` classes for the batch
            sampled = tf.random_shuffle(tf.range(num_classes))[:num_classes_per_batch]
            # Repeat each element `num_images_per_class` times
            batch_labels = tf.tile(tf.expand_dims(sampled, -1), [1, num_utt_per_class])
            return tf.to_int64(tf.reshape(batch_labels, [-1]))
        selector = tf.contrib.data.Counter().map(generator)
        selector = selector.apply(tf.contrib.data.unbatch())
        dataset = tf.contrib.data.choose_from_datasets(datasets, selector)

        # Batch
        batch_size = num_classes_per_batch * num_utt_per_class
        dataset = dataset.batch(batch_size)

        self.dataset = dataset.make_one_shot_iterator()

    def parse(self, proto):
        if len(self.data_shape) == 0:
            keys_to_features = {
                'data': tf.VarLenFeature(dtype=tf.float32),
                'label': tf.FixedLenFeature(shape=(1), dtype=tf.int64),
            }
            parsed_ = tf.parse_single_example(proto, keys_to_features)
            #data = tf.reshape(parsed_['data'], [-1, self.feature_dim])
        else:
            keys_to_features = {
                'data': tf.FixedLenFeature(shape=self.data_shape, dtype=tf.float32),
                'label': tf.FixedLenFeature(shape=(1), dtype=tf.int64),
            }
            parsed_ = tf.parse_single_example(proto, keys_to_features)
        data = parsed_['data']
        #data = tf.sparse_tensor_to_dense(data)
        #data = tf.reshape(data, self.data_shape)
        #data = tf.RaggedTensor.from_tensor(data, padding=0)
        return data, parsed_['label']

    def get_next(self):
        return self.dataset.get_next()