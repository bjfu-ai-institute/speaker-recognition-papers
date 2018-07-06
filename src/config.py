import h5py


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
        save_path=None
    ):
        if config_path:
            f = h5py.File(config_path)
            self = f['config']
            print("successfully load " + f[name])
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
    
    def set(self,
        n_speaker=None,
        batch_size=None,
        n_gpu=None,
        max_step=None,
        is_big_dataset=None,
        url_of_bigdataset_temp_file=None,
        learning_rate=None,
        save_path=None):
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

    def write(self, path, name='global config'):
        f = h5py.File(path, 'w')
        f.create_dataset('config', data=self)
        f.create_dataset('name', data=name)
