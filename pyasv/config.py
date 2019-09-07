import yaml
import os
import logging
import time


class Config:
    def __init__(self, config):
        self.read_yaml(config)
        self.defaults = self.get_defaults()
        self._set_project_loggers()

    def set_value(self, **kwargs):
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]

    def save(self, name='config.yaml'):
        with open(name, 'w') as f:
            dic = self.__dict__
            yaml.dump(dic, f, indent=4)

    def _set_project_loggers(self):
        formatter = logging.Formatter('%(asctime)s [%(filename)s: %(lineno)d] %(levelname)s: %(message)s',
                                      '%m-%d %H:%M:%S')
        self._single_logger('data', formatter)
        self._single_logger('train', formatter)

    def _single_logger(self, name, formatter):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.save_path + '/log/%s.txt' % name)
        # ch = logging.StreamHandler()
        fh.setFormatter(formatter)
        # ch.setFormatter(formatter)
        logger.addHandler(fh)
        # logger.addHandler(ch)

    def get(self, dic, name):
        try:
            res = dic[name]
        except KeyError:
            res = self.defaults[name]
        return res

    def read_yaml(self, file_path):
        with open(file_path, 'r') as f:
            dic = yaml.load(f, Loader=yaml.Loader)
            self.lr = self.get(dic, 'lr')
            self.max_step = self.get(dic, 'max_step')
            self.save_path = self.get(dic, 'save_path')
            self.model_name = self.get(dic, 'model_name')
            self.feature_dims = self.get(dic, 'feature_dims')
            self.n_gpu = self.get(dic, 'n_gpu')
            self.n_threads = self.get(dic, 'n_threads')
            self.n_speaker = self.get(dic, 'n_speaker')
            self.slides = self.get(dic, 'slides')
            self.sample_rate = self.get(dic, 'sample_rate')
            self.fix_len = self.get(dic, 'fix_len')
            self.num_utt_per_class = self.get(dic, 'num_utt_per_class')
            self.num_classes_per_batch = self.get(dic, 'num_classes_per_batch')
            self.batch_nums_per_epoch = self.get(dic, 'batch_nums_per_epoch')
            self.n_fft = self.get(dic, 'n_fft')
            self.n_speaker_test = self.get(dic, 'n_speaker_test')
            self.hop_length = self.get(dic, 'hop_length')

    def get_defaults(self):
        dic = {}
        dic['max_step'] = 100
        dic['save_path'] = os.path.abspath("~/save/")
        dic['model_name'] = 'model'
        dic['feature_dims'] = 40
        dic['n_gpu'] = 1
        dic['n_threads'] = 8
        dic['n_speaker'] = 0
        dic['slides'] = [0, 0]
        dic['sample_rate'] = 16000
        dic['fix_len'] = 8 # second
        dic['num_utt_per_class'] = 4
        dic['num_classes_per_batch'] = 16
        dic['batch_nums_per_epoch'] = 1200
        dic['n_fft'] = 512
        dic['n_speaker_test'] = 0
        dic['hop_length'] = 512