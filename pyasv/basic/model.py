import tensorflow as tf
from pyasv.basic import utils
import os


class Model(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, x, y, training=True):
        if training:
            self._build_train_graph()
        else:
            self._build_test_graph()
        pass

    def inference(self, x):
        pass

    def _build_train_graph(self):
        pass

    def _build_test_graph(self):
        pass

    def get_tensor(self, name):
        try:
            tensor = tf.get_default_graph().get_tensor_by_name(name)
        except ValueError:
            try:
                tensor = tf.get_default_graph().get_operation_by_name(name).outputs
                if len(tensor) == 1:
                    tensor = tensor[0]
            except KeyError:
                tensor = None
        return tensor


    def create_url(self, urls, enroll=None, test=None):
        ids = 0
        spk2id_train = {}
        id2utt_train = {}
        count = 0
        for url in urls:
            with open(url, 'r') as f:
                datas = f.readlines()
            for line in datas:
                p, spk = line.replace("\n", "").split(' ')
                if spk not in spk2id_train.keys():
                    spk2id_train[spk] = ids
                    id2utt_train[spk2id_train[spk]] = []
                    ids += 1
                id2utt_train[spk2id_train[spk]].append(p)
            utils.write_dict_to_text(os.path.join(self.config.save_path, 'url',
                                                  "train_tmp_%d" % count), id2utt_train)
            for key in spk2id_train.keys():
                id2utt_train[spk2id_train[key]] = []
            count += 1
        ids = 0
        spk2id_test = {}
        id2utt_test = {}
        if enroll is not None:
            with open(enroll, 'r') as f:
                datas = f.readlines()
            for line in datas:
                url, spk = line.replace("\n", "").split(' ')
                if spk not in spk2id_test.keys():
                    spk2id_test[spk] = ids
                    id2utt_test[spk2id_test[spk]] = []
                    ids += 1
                id2utt_test[spk2id_test[spk]].append(url)
            utils.write_dict_to_text(os.path.join(self.config.save_path,
                                                  'url', "enroll_tmp"), id2utt_test)
            for key in spk2id_test.keys():
                id2utt_test[spk2id_test[key]] = []
        if test is not None:
            with open(test, 'r') as f:
                datas = f.readlines()
            for line in datas:
                url, spk = line.replace("\n", "").split(' ')
                id2utt_test[spk2id_test[spk]].append(url)
            utils.write_dict_to_text(os.path.join(self.config.save_path, 'url', "test_tmp"), id2utt_test)
            return len(id2utt_train.keys()), len(id2utt_test.keys())