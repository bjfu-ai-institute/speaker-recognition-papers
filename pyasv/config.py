import yaml
import os
import logging
import time


class Config:
    def __init__(self, config):
        self.read_yaml(config)
        name = time.strftime("backup-%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        for dd in ['log', 'graph', 'model']:
            if os.path.exists(os.path.join(self.save_path, dd)):
                try:
                    os.mkdir(os.path.join(self.save_path, name))
                except:
                    pass
                os.rename(os.path.join(self.save_path, dd),
                          os.path.join(self.save_path, name, dd))
                logging.info('Moving %s to backup' % dd)
            os.mkdir(os.path.join(self.save_path, dd))
        self._set_project_loggers()

    def set_value(self, **kwargs):
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]

    def save(self, name='config.yaml'):
        with open(os.path.join(self.save_path, name), 'w') as f:
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
        fh = logging.FileHandler(self.save_path+'/log/%s.txt' % name)
        #ch = logging.StreamHandler()
        fh.setFormatter(formatter)
        #ch.setFormatter(formatter)
        logger.addHandler(fh)
        #logger.addHandler(ch)

    def read_yaml(self, file_path):
        with open(file_path, 'r') as f:
            dic = yaml.load(f)
            self.lr = dic['lr']
            self.max_step = dic['max_step']
            self.save_path: [str, bool] = dic['save_path']
            self.model_name: [str, bool] = dic['model_name']
            self.feature_dims: [int, bool] = dic['feature_dims']
            self.n_gpu: int = dic['n_gpu']
            self.n_threads = dic['n_threads']
            self.n_speaker = dic['n_speaker']
            self.slides = dic['slides']
            self.sample_rate: int = dic['sample_rate']
            self.fix_len: int = dic['fix_len']
            self.num_utt_per_class = dic['num_utt_per_class']
            self.num_classes_per_batch = dic['num_classes_per_batch']
