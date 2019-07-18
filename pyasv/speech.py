import librosa
import time
import numpy as np
import logging
import os
from pyasv import ops
from pyasv import TFrecordGen
from pyasv import TFrecordClassBalanceGen


def slide_windows(feature, slide_windows):
    if type(feature)!=np.ndarray:
        feature = np.array(feature)
    result_ = []
    if slide_windows is not None:
        l, r = slide_windows
        for i in range(feature.shape[0]-r):
            if i < l:
                continue
            else:
                result_.append(np.array(feature[i-l:i+r+1]))
        result = np.array(result_)
    else:
        result = feature
    return result


def add_wave_header():
    return


class FilterBank:
    def __init__(self, url_folder, config, data_type=None):
        self.config = config
        self.n_fft = 512 if config.n_fft is None else config.n_fft
        self.process_num = 10 if config.n_threads is None else config.n_threads
        self.fix_len = True if config.fix_len is None else config.fix_len
        self.dims = config.feature_dims
        self.sample_rate = config.sample_rate
        self.slides = [None, None] if config.slides is None else config.slides
        self.logger = logging.getLogger("data")
        self.url_folder = url_folder
        self.data_type = data_type
    
    def _extract_one(self, url):
        y, sr = librosa.load(url, sr=self.sample_rate)
        fbank = librosa.feature.melspectrogram(y, sr=sr, n_fft=self.n_fft, n_mels=self.dims).T
        return librosa.power_to_db(fbank, ref=np.max)
    
    def extract(self, url_list):
        labels = [item[1] for item in url_list]
        url_list = [item[0] for item in url_list]
        fbanks = ops.multi_processing(self._extract_one, url_list, self.process_num)
        if self.slides != [None, None]:
            slides = [self.slides for i in range(len(fbanks))]
            fbanks = ops.multi_processing(slide_windows, zip(fbanks, slides), self.process_num)
        return fbanks, labels

    def extract_rcd(self):
        train_gen = TFrecordClassBalanceGen(self.config, 'train')
        train_urls, enroll_urls, test_urls = self.read_url_file()
        for i in range(len(train_urls)):
            data, label = self.extract(train_urls)
            train_gen.write(data, label)
            del data, label

    def read_url_file(self):
        files = os.listdir(self.url_folder)
        train_urls, enroll_urls, test_urls = [], [], []
        for f in files:
            if f[:5] == 'train':
                with open(os.path.join(self.url_folder, f)) as data:
                    train_urls.append(list(map(lambda x: (str(x.split(' ')[0]), int(x.split(' ')[1])),
                                               data.read().splitlines())))
            elif f[:4] == 'test':
                with open(os.path.join(self.url_folder, f)) as data:
                    enroll_urls = list(map(lambda x: (str(x.split(' ')[0]), int(x.split(' ')[1])),
                                       data.read().splitlines()))
            elif f[:6] == 'enroll':
                with open(os.path.join(self.url_folder, f)) as data:
                    test_urls = list(map(lambda x: (str(x.split(' ')[0]), int(x.split(' ')[1])),
                                       data.read().splitlines()))
            else: raise NameError("Url file should be [train*] [enroll*] [test*] contain all audio file path.")
        return train_urls, test_urls, enroll_urls

    def extract_class_balance_rcd(self, file_path):
        pass
