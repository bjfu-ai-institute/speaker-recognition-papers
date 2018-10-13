import json
import os
import logging


__TrainError__ = ['lr', 'max_step', 'batch_size', 'batch_nums_per_epoch']
__SystemError__ = ['save_path', 'n_threads', 'model_name']
__DataError__ = ['feature_dims', 'n_speaker', 'batch_size']

__SystemWarning__ = ['n_gpu']
__DataWarning__ = ['slides']


class Config:
    def __init__(self, config, **kwargs):
        with open(config, 'r') as f:
            dic = json.load(f)
            self.__dict__ = dic
        self.__init_infolist()
        self._check_error()
        self._check_warning()
        self._set_log()

    def __init_infolist(self):
        self.error = []
        self.warning = []

    def _check_error(self):
        if 'model_name' not in self.__dict__.keys():
            self.model_name = 'model'
        logger = logging.getLogger(self.model_name)
        exit_status = False
        for key in self.error:
            if key not in self.__dict__.keys():
                logger.error('Missing parameters %s' % key)
                exit_status = True
        if exit_status:
            exit()

    def _check_warning(self):
        logger = logging.getLogger(self.model_name)
        for key in self.warning:
            if key not in self.__dict__.keys():
                logger.warning('Missing parameters %s' % key)

    def save(self, name='config.json'):
        """This method is used for save your config to save_path

        Parameters
        ----------
        name : ``str``.
            the name of your config file.
        """
        with open(os.path.join(self.save_path, name), 'w') as f:
            dic = self.__dict__
            json.dump(dic, f, indent=4)

    def _set_log(self):
        logger = logging.getLogger('%s'%self.model_name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('log_%s.txt'%self.model_name)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(filename)s line:%(lineno)d]-%(levelname)s: %(message)s',
                                      '%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)


class FeatureConfig(Config):
    def __init__(self, config_path):
        super().__init__(config_path)

    def __init_infolist(self):
        self.error = __DataError__ + __SystemError__
        self.warning = __DataWarning__


class TrainConfig(Config):
    def __init__(self, config_path):
        super().__init__(config_path)

    def __init_infolist(self):
        self.error = __DataError__ + __SystemError__ + __TrainError__
        self.warning = __DataWarning__ + __SystemWarning__
