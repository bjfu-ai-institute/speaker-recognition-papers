import logging
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim


class TFrecordGen:
    def __init__(self, config):
        self.log = logging.getLogger(config.model_name)
        self.url = config.save_path
        self.num_threads = config.n_threads

    def write(self, data, label, name):
        output_file = os.path.join(self.url, name)
        self.log.info("Writing %s, received data shape is %s, label shape is %s"%(output_file,
                                                                                  np.array(data).shape,
                                                                                  np.array(label).shape))
        with tf.python_io.TFRecordWriter(output_file) as writer:
            for i, j in zip(data, label):
                feature_dict = {
                    'data': tf.train.Feature(float_list=tf.train.FloatList(value=np.array(i).reshape(-1,))),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(j).reshape(-1,)))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(example.SerializeToString())
        self.log.info("Writing %s finished." % output_file)


class TFrecordReader:
    def __init__(self, files, data_shape, label_shape, descriptions=None):
        self.data_files = files
        self.keys_to_features = {
            'data': tf.FixedLenFeature(shape=data_shape, dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=label_shape, dtype=tf.int64)
        }
        self.items_to_handlers = {
            'data': slim.tfexample_decoder.Tensor('data'),
            'label': slim.tfexample_decoder.Tensor('label')
        }
        self.items_to_descriptions = descriptions
        # {
        #    'data' : 'a 2X2X2 int64 array',
        #    'label' : 'a 2X2X2 float32 array'
        # }

    def _parse(self, proto):
        parsed_ = tf.parse_single_example(proto, self.keys_to_features)
        return parsed_['data'], parsed_['label']

    def read(self, batch_size, repeat=True, shuffle=False):
        dataset = tf.data.TFRecordDataset(self.data_files)
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.map(self._parse)
        dataset = dataset.batch(batch_size)
        return dataset
