import json
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
                 bn_epsilon=None,
                 slide_windows=None,
                 plda_rankf=None,
                 plda_rankg=None,
                 deep_speaker_out_channel=None,
                 audio_n_filt=None):
        """
        Parameters
        ----------
        config_path : ``str``
            The path of config which you want to restore,
            If you want to create a new config, this param should
            be None.
        is_big_dataset : ``bool``
            If we store the features to disk and restore a part of feature each step.
        url_of_bigdataset_temp_file : ``str``
            To decide the save path of the features you want to restore.
            If the param "is_big_dataset" is False, this param should be None.
        slide_windows : ``list``
            This list have two element. The first is `l` and the second is `r`
            If slide_windows is not ``None``, each frame's feature will be replace by [i-l, i+r] frames' feature.
        """
        if config_path:
            f = open(config_path, 'r')
            dic = json.load(f)
            self.__dict__ = dic

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
            self.SLIDE_WINDOWS = slide_windows
            self.PLDA_F_RANK = plda_rankf
            self.PLDA_G_RANK = plda_rankg
            self.DEEP_SPEAKER_OUT_CHANNEL = deep_speaker_out_channel
            self.Audio_n_filt = audio_n_filt

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
            bn_epsilon=None,
            slide_windows=None,
            plda_rankf=None,
            plda_rankg=None,
            deep_speaker_out_channel=None,
            audio_n_filt=None):
        """The ``set`` method is used for reset some config.
        """
        if n_speaker is not None:
            self.N_SPEAKER = n_speaker
        if batch_size is not None:
            self.BATCH_SIZE = batch_size
        if n_gpu is not None:
            self.N_GPU = n_gpu
        if max_step is not None:
            self.MAX_STEP = max_step
        if is_big_dataset is not None:
            self.IS_BIG_DATASET = is_big_dataset
        if url_of_bigdataset_temp_file is not None:
            self.URL_OF_BIG_DATASET = url_of_bigdataset_temp_file
        if learning_rate is not None:
            self.LR = learning_rate
        if save_path is not None:
            self.SAVE_PATH = save_path
        if conv_weight_decay is not None:
            self.CONV_WEIGHT_DECAY = conv_weight_decay
        if fc_weight_dacay is not None:
            self.FC_WEIGHT_DECAY = fc_weight_dacay
        if bn_epsilon is not None:
            self.BN_EPSILON = bn_epsilon
        if slide_windows is not None:
            self.SLIDE_WINDOWS = slide_windows
        if plda_rankf is not None:
            self.PLDA_F_RANK = plda_rankf
        if plda_rankg is not None:
            self.PLDA_G_RANK = plda_rankg
        if deep_speaker_out_channel is not None:
            self.DEEP_SPEAKER_OUT_CHANNEL = deep_speaker_out_channel
        if audio_n_filt is not None:
            self.Audio_n_filt = audio_n_filt

    def save(self, name='global_config'):
        """This method is used for save your config to save_path

        Parameters
        ----------
        name : ``str``.
            the name of your config file.
        """
        f = open(os.path.join(self.SAVE_PATH, name+'.json'), 'w')
        dic = self.__dict__
        json.dump(dic, f)
        f.close()
