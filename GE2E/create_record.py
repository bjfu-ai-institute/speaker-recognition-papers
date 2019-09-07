from pyasv.pipeline import TFrecordClassBalanceGen
from pyasv.config import Config
from pyasv.speech_processing import ext_fbank_feature
import os
from pyasv.basic import ops
import numpy as np

enroll_url = '/home/data/speaker-recognition/url/enroll'
test_url = '/home/data/speaker-recognition/url/test'


def write_dict_to_text(path, dic, key_before_value=True, data_one_line=False, spaced=" "):
    with open(path, 'w') as f:
        for key in dic.keys():
            if type(dic[key]) == list or type(dic[key]) == set:
                if data_one_line:
                    data = "%s " % key
                    for dat in dic[key]:
                        data += dat + spaced
                    f.writelines(data + '\n')
                else:
                    for dat in dic[key]:
                        data = "%s %s\n" % (dat, key)
                        f.writelines(data)


def create_url(config, urls, enroll=None, test=None):
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
        write_dict_to_text(os.path.join(config.save_path, 'url', "train_tmp_%d" % count), id2utt_train)
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
        write_dict_to_text(os.path.join(config.save_path, 'url', "enroll_tmp"), id2utt_test)
        for key in spk2id_test.keys():
            id2utt_test[spk2id_test[key]] = []
    if test is not None:
        with open(test, 'r') as f:
            datas = f.readlines()
        for line in datas:
            url, spk = line.replace("\n", "").split(' ')
            id2utt_test[spk2id_test[spk]].append(url)
            write_dict_to_text(os.path.join(config.save_path, 'url', "test_tmp"), id2utt_test)
        return len(id2utt_train.keys()), len(id2utt_test.keys())


def limit_len(data):
    while data.shape[0] < 150:
        data = np.concatenate((data, np.zeros(shape=(data.shape[0],))), 0)
    data = data[:100]
    return data


def run(config):
    reader = TFrecordClassBalanceGen(config, 'train')
    train_urls = [os.path.join(config.save_path, 'url/train_tmp_0')]
    for train_url in train_urls:
        x, y = ext_fbank_feature(train_url, config)
        x = ops.multi_processing(limit_len, x, config.n_threads, True)
        reader.write(x, y)
        del x, y
