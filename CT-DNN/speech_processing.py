import librosa
import os
import numpy as np
import config


def slide_windows(feature):
    """
    Make feature[i] contain [i-4:i+4] frames' features.
    """
    if type(feature)!=np.ndarray:
        feature = np.array(feature)
    result_ = []
    for i in range(feature.shape[0]-5):
        if i < 4:
            continue
        else:
            result_.append(np.array(feature[i-4:i+5]))
    return np.array(result_)
        

def ext_mfcc_feature(url_path):
    """
    Return the MFCC feature
    """
    with open(url_path, 'r') as urls:
        mfccs = []
        labels = []
        for url in list(urls):
            url, label = str(url).split(" ")
            index = eval(str(label).split("\n")[0])
            label = np.zeros(config.N_SPEAKER)
            label[index] = 1
            y, sr = librosa.load(url)
            mfcc_ = librosa.feature.mfcc(y, sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc_, width=3)
            mfcc_delta_delta = librosa.feature.delta(mfcc_delta, width=3)
            mfcc = np.vstack([mfcc_, np.vstack([mfcc_delta, mfcc_delta_delta])])
            mfcc = slide_windows(mfcc)
            for i in mfcc:
                mfccs.append(i)
                labels.append(label)
        return mfccs, labels


def ext_fbank_feature(url_path):
    """
    Return the fbank feature
    """
    with open(url_path, 'r') as urls:
        fbanks = []
        labels = []
        for url in list(urls):
            url, label = str(url).split(" ")
            index = eval(str(label).split("\n")[0])
            label = np.zeros(config.N_SPEAKER)
            label[index] = 1
            y, sr = librosa.load(url)
            fbank = librosa.feature.melspectrogram(y, sr)
            fbank = slide_windows(fbank)
            for i in fbank:
                fbanks.append(i)
                labels.append(label)
        return fbanks, labels
