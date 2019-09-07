import logging
import os
import tensorflow as tf
import numpy as np
import collections
import threading


class Writer:
    def __init__(self, file_name):
        self.file_name = file_name
        self.write_count = 0
        self.writer = tf.python_io.TFRecordWriter(file_name)

    def __str__(self):
        return "%d data(s) has been written into %s"%(self.write_count, self.file_name)

    def write(self, example):
        self.write_count += 1
        self.writer.write(example)


class TFrecordGen:
    def __init__(self, config, file_name):
        self.log = logging.getLogger('data')
        self.url = config.save_path
        self.num_threads = config.n_threads
        self.file_name = file_name
        self.writer = Writer(os.path.join(self.url, 'data', self.file_name))

    def write(self, datas, keys):
        output_file = os.path.join(self.url, 'data', self.file_name)
        if datas[0].shape[0] != 0:
            self.log.info("Writing to %s, received data number is %s"%(output_file, datas[0].shape[0]))
        feature_list = {}
        for i, key in enumerate(keys, start=0):
            #feature_dict = {
            #    'data': tf.train.Feature(float_list=tf.train.FloatList(value=np.array(i).reshape(-1,))),
            #    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(j).reshape(-1,)))
            #}
            feature_list[key] = datas[i]
        for i in range(datas[0].shape[0]):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={key: tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=feature_list[key][i].reshape(-1))) for key in keys}))
            self.writer.write(example.SerializeToString())
        self.log.info(str(self.writer))

    def close(self):
        self.writer.close()


class TFrecordReader:
    def __init__(self, files, keys_to_features):
        """
        :param keys_to_features:
            :type OrderedDict { "key": tf.FixedLenFeature() }
        """
        assert isinstance(keys_to_features, collections.OrderedDict)
        assert isinstance(files, list)
        self.data_files = files
        self.keys_to_features = keys_to_features

    def parse(self, proto):
        parsed_ = tf.parse_single_example(proto, self.keys_to_features)
        key_list = self.keys_to_features.keys()
        return_tuple = tuple(parsed_[key] for key in key_list)
        return return_tuple

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
        self.writer = [TFrecordGen(config, file_name+'%d.crcd'%i) for i in range(self.class_num)]

    def write(self, datas, keys, label):
        if len(label.shape) == 2 and label.shape[-1] == 1:
            label = label.reshape(-1,)
        elif len(label.shape) == 1:
            pass
        else:
            raise ValueError("label's shape must be 1-d array or 2-d array which like (x, 1)")
        # [spkr_num, data_num]
        datas = list(list(data[np.where(label == i)[0]] for data in datas) for i in range(self.class_num))
        for i in range(len(datas)):
            self.writer[i].write(datas[i], keys)

    def close(self):
        for i in self.writer:
            self.i.close()


class TFrecordClassBalanceReader:
    def __init__(self, config, filenames, keys_to_features):
        """
            keys_to_features = {
                'data': tf.VarLenFeature(dtype=tf.float32),
                'label': tf.FixedLenFeature(shape=(1), dtype=tf.int64),
            }

        """
        self.keys_to_features = keys_to_features
        assert isinstance(keys_to_features, collections.OrderedDict)

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
        assert isinstance(keys_to_features, collections.OrderedDict)
        # Batch
        batch_size = num_classes_per_batch * num_utt_per_class
        dataset = dataset.batch(batch_size)
        self.dataset = dataset.make_one_shot_iterator()

    def parse(self, proto):
        parsed_ = tf.parse_single_example(proto, self.keys_to_features)
        keys = self.keys_to_features.keys()
        return list(parsed_[key] for key in keys)

    def get_next(self):
        return self.dataset.get_next()
