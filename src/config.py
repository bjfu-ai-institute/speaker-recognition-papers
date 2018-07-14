import pickle as pkl


class Config:
    def __init__(
        self,
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
        fc_weight_dacay=None,
        bn_epsilon=None):
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
            self.FC_WEIGHT_DECAY = fc_weight_dacay
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
        
    def save(self, path, name='global config'):
        f = open(path+'.pkl' , 'wb+')
        dic = {'config':self, 'name':name}
        pkl.dump(dic, f)
        f.close()
        