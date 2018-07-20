import pickle as pkl
import os


class Config:

    def __init__(self,
                 config_path=None,
                 name=None,
                 n_speaker=None,
                 batch_size=None,
                 n_gpu=None,
                 max_step=None,
                 is_big_dataset=None,
                 url_of_bigdataset_temp_file=None,
                 learning_rate=None,
                 save_path=None,
                 conv_weight_decay=None,
                 fc_weight_decay=None,
                 bn_epsilon=None):
        """
        :param: config_path: The path of config which you want to restore,
                             If you want to create a new config, this param should
                             be None.
        :param is_big_dataset: If we store the features to disk and restore a part of feature each step.
                               The type is bool
        :param url_of_bigdataset_temp_file: To decide the save path of the features you want to restore.
                                            If the param "is_big_dataset" is False, this param should be None.
        """
        if config_path:
            f = open(config_path, 'rb')
            dic = pkl.load(f)
            self = dic['config']
            print("successfully load " + dic[name])
        else:
            self.BATCH_SIZE = batch_size
            self.N_GPU = n_gpu
            self.MODEL_NAME = name
            self.N_SPEAKER = n_speaker
            self.MAX_STEP = max_step
            self.IS_BIG_DATASET = is_big_dataset
            if self.IS_BIG_DATASET:
                self.URL_OF_BIG_DATASET = url_of_bigdataset_temp_file
            self.LR = learning_rate
            self.SAVE_PATH = save_path
            self.CONV_WEIGHT_DECAY = conv_weight_decay
            self.FC_WEIGHT_DECAY = fc_weight_decay
            self.BN_EPSILON = bn_epsilon
    
    def set(self,
            n_speaker=None,
            batch_size=None,
            n_gpu=None,
            max_step=None,
            is_big_dataset=None,
            url_of_bigdataset_temp_file=None,
            learning_rate=None,
            save_path=None,
            conv_weight_decay=None,
            fc_weight_dacay=None,
            bn_epsilon=None):
        """The ``set`` method is used for reset some config.
        """
        if n_speaker:
            self.N_SPEAKER = n_speaker
        if batch_size:
            self.BATCH_SIZE = batch_size
        if n_gpu:
            self.N_GPU = n_gpu
        if max_step:
            self.MAX_STEP = max_step
        if is_big_dataset:
            self.IS_BIG_DATASET = is_big_dataset
        if url_of_bigdataset_temp_file:
            self.URL_OF_BIG_DATASET = url_of_bigdataset_temp_file
        if learning_rate:
            self.LR = learning_rate
        if save_path:
            self.SAVE_PATH = save_path
        if conv_weight_decay:
            self.CONV_WEIGHT_DECAY = conv_weight_decay
        if fc_weight_dacay:
            self.FC_WEIGHT_DECAY = fc_weight_dacay
        if bn_epsilon:
            self.BN_EPSILON = bn_epsilon
        
    def save(self, name='global_config'):
        """This method is used for save your config to save_path

        :param name: ``str``.
            the name of your config file.
        """
        f = open(os.path.join(self.SAVE_PATH, name+'.pkl'), 'wb+')
        dic = {'config': self, 'name': name}
        pkl.dump(dic, f)
        f.close()
