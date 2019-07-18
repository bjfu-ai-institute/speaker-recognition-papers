"""
ext_mfcc_feature
----------------
.. autofunction:: pyasv.speech_processing.ext_mfcc_feature

ext_fbank_feature
-----------------

.. autofunction:: pyasv.speech_processing.ext_fbank_feature

.. note::
    The file contain the path to all audio and its number in the dataset.

    The contents of the file should be as follows:

    xxxxx/your_data_path/1_1.wav 0

    xxxxx/your_data_path/1_2.wav 0

    xxxxx/your_data_path/2_1.wav 1

calc_fbank
----------

.. autofunction:: pyasv.speech_processing.calc_fbank


calc_cqcc
---------

.. autofunction:: pyasv.speech_processing.calc_cqcc


slide_windows
-------------

.. autofunction:: pyasv.speech_processing.slide_windows


cqcc_resample
-------------

.. autofunction:: pyasv.speech_processing.cqcc_resample


cmvn
----

.. autofunction:: pyasv.speech_processing.cmvn

"""
import librosa
import time
import numpy as np
import logging
import scipy
import resampy
from pyasv.basic import ops
from scipy.io.wavfile import read
from scipy.fftpack import dct


def slide_windows(feature, slide_windows):
    """concat the feature with the frame before and after it.

    Parameters
    ----------
    feature : ``list`` or ``np.ndarray``
        feature array of an audio file.

    Returns
    -------
    result : ``list`` or ``np.ndarray``
        the feature array after concat. The type is same as input.
    """
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


def ext_mfcc_feature(url_path, config):
    """This function is used for extract MFCC feature of a dataset.

    Parameters
    ----------
    url_path : ``str``
        The path of the 'PATH' file.
    config : ``config``
        config of feature. (To decide if we need slide_window, and params of slide_window)

    Returns
    -------
    fbank : ``list``
        The feature array. each frame concat with the frame before and after it.
    label : ``list``
        The label of fbank feature.

    """
    logger = logging.getLogger('data')

    with open(url_path, 'r') as urls:
        labels = []
        url_list = []
        for url in list(urls):
            line, label = str(url).split(" ")
            index = eval(str(label).split("\n")[0])
            labels.append([index])
            url_list.append(line)

    logger.info("Extracting MFCC feature, utt_nums is %d" % len(url_list))

    n_filt = [config.feature_dims for i in range(len(url_list))]
    slides = [config.slides for i in range(len(url_list))]
    if hasattr(config, 'min_db'):
        dbs = [config.min_db for i in range(len(url_list))]
        mfccs = ops.multi_processing(calc_mfcc, zip(url_list, n_filt, slides, dbs), config.n_threads)
    else:
        mfccs = ops.multi_processing(calc_mfcc, zip(url_list, n_filt, slides), config.n_threads)
    logger.info("Extracting MFCC feature succeed")
    return mfccs, labels


def calc_mfcc(url, n_filt, slide_l, slide_r, min_db=None):
    y, sr = librosa.load(url)
    if min_db is not None:
        y = naive_vad(y, min_db)
    mfcc_ = librosa.feature.mfcc(y, sr, n_mfcc=n_filt)
    mfcc_delta = librosa.feature.delta(mfcc_, width=3)
    mfcc_delta_delta = librosa.feature.delta(mfcc_delta, width=3)
    mfcc = np.vstack([mfcc_, np.vstack([mfcc_delta, mfcc_delta_delta])])
    mfcc = cmvn(mfcc)
    if (slide_r and slide_l) is not None:
        slides = [slide_l, slide_r]
        mfcc = slide_windows(mfcc, slides)
    return mfcc


def naive_vad(y, min_db):
    points = librosa.effects.split(y, top_db=min_db)
    y_ = None
    for i, j in zip(points):
        if y_ is None:
            y_ = y[i:j]
        else:
            y_ = np.concatenate(y_, y[i:j])
    return y_


def ext_fbank_feature(url_path, config):
    """This function is used for extract features of one dataset.

    Parameters
    ----------
    url_path : ``str``
        The path of the 'PATH' file.
    config : ``config``
        config of feature. (Contain the parameters of slide_window and feature_dims)

    Returns
    -------
    fbank : ``list``
        The feature array. each frame concat with the frame before and after it.
    label : ``list``
        The label of fbank feature.

    Notes
    -----
    Changeable concat size is in the todolist

    """
    logger = logging.getLogger('data')

    url_list = []
    labels = []
    with open(url_path, 'r') as urls:
        for line in list(urls):
            url, label = str(line).split(" ")
            index = eval(str(label).split("\n")[0])
            url_list.append(url)
            labels.append([index])

    logger.info("Extracting fbank feature, utt_nums is %d" % len(url_list))
    if config.fix_len is not None:
        max_len = get_max_audio_time(url_list)
        max_lens = [max_len for i in range(len(url_list))]
    else:
        max_lens = [None for i in range(len(url_list))]
    n_filt = [config.feature_dims for i in range(len(url_list))]
    if config.slides is not None:
        slide_l = [config.slides[0] for i in range(len(url_list))]
        slide_r = [config.slides[1] for i in range(len(url_list))]
    else:
        slide_l = [None for i in range(len(url_list))]
        slide_r = [None for i in range(len(url_list))]

    fbanks = ops.multi_processing(calc_fbank, zip(url_list, n_filt, slide_l, slide_r, max_lens), config.n_threads)

    logger.info("Extracting fbank feature succeed")
    return fbanks, labels


def get_max_audio_time(url_list):
    max_len = 0
    for i in url_list:
        y, _ = librosa.load(i)
        if y.shape[0] > max_len:
            max_len = y.shape[0]
    return max_len


def calc_fbank(url, n_filt, slide_l=None, slide_r=None, max_len=None):
    """Calculate Fbank feature of a audio file.

    Parameters
    ----------
    url : ``str``
        Path to the audio file.

    Returns
    -------
    fbank : ``np.ndarray``
        Fbank feature of this audio.
    """
    sample_rate, signal = read(url)
    y, sr = librosa.load(url)
    if max_len is not None:
        y = librosa.util.fix_length(y, max_len)
    filter_banks = librosa.feature.melspectrogram(y, sr, n_fft=512, n_mels=n_filt).T

    if slide_l == 0 and slide_r == 0:
        slide_l = None
        slide_r = None
    if (slide_r and slide_l) is not None:
        slides = [slide_l, slide_r]
        filter_banks = slide_windows(filter_banks, slides)
    return filter_banks


def calc_cqcc(url):
    """Calculate CQCC feature of a audio file.

    Parameters
    ----------
    url : ``str``
        path ot the audio file.

    Returns
    -------
    cqcc : ``np.ndarray``
        CQCC feature.
    """
    y, sr = librosa.load(url)
    constant_q = librosa.cqt(y=y, sr=sr)
    cqt_abs = np.abs(constant_q)
    cqt_abs_square = cqt_abs ** 2
    cqt_spec = librosa.amplitude_to_db(cqt_abs_square).astype('float32')
    cqt_resampy_spec = cqcc_resample(cqt_spec, sr, 44000)
    cqcc = scipy.fftpack.dct(cqt_resampy_spec, norm='ortho', axis=0)
    return cqcc


def cqcc_resample(s, fs_orig, fs_new, axis=0):
    """implement the resample operation of CQCC

    Parameters
    ----------
    s : ``np.ndarray``
        the input spectrogram.
    fs_orig : ``int``
        origin sample rate
    fs_new : ``int``
        new sample rate
    axis : ``int``
        the resample axis

    Returns
    -------
    spec_res : ``np.ndarray``
        spectrogram after resample
    """
    if int(fs_orig) != int(fs_new):
        s = resampy.resample(s, sr_orig=fs_orig, sr_new=fs_new,
                             axis=axis)
    return s


def cmvn(feature):
    """Apply cmvn to feature list/array.

    Parameters
    ----------
    feature : ``np.ndarray``

    returns
    -------
    feature_list : ``np.ndarray``
        the feature after cmvn.

    Notes
    -----
    We have used `cmvn` while calculating mfcc or fbank.
    """
    if type(feature) == np.ndarray:
        feature = np.array(feature)

    N = feature.shape[0]
    mean_m = np.sum(feature, axis=0) / N
    std_m = np.sqrt(np.sum(feature ** 2, axis=0) / N - mean_m ** 2)
    for col in range(feature.shape[1]):
        if std_m[col] != 0:
            feature[:, col] = (feature[:, col] - mean_m[col]) / std_m[col]
        else:
            feature[:, col] = feature[:, col] - mean_m[col]
    return feature


def get_stft(path=None, y=None, sr=None, NFFT=None, frame_size=None):
    if frame_size is None:
        frame_size = 0.025
    if NFFT is None:
        NFFT = 1024
    if (path is None and y is None) or (path is not None and y is not None):
        raise ValueError("Determine path or y")
    if path is not None:
        y, sr = librosa.load(path)
    fft_res = librosa.stft(y, n_fft=NFFT, hop_length=int(frame_size*sr))[:, :(NFFT//2 + 1)]
    mag_spec = np.abs(fft_res)
    phase_spec = np.angle(fft_res)
    return mag_spec.T, phase_spec.T


def ext_spec_feature(url_path, config):
    logger = logging.getLogger('data')

    with open(url_path, 'r') as urls:
        labels = []
        url_list = []
        for url in list(urls):
            line, label = str(url).split(" ")
            index = eval(str(label).split("\n")[0])
            labels.append([index])
            url_list.append(line)

    logger.info("Extracting Spec feature, utt_nums is %d" % len(url_list))
    NFFT = [config.NFFT for i in range(len(url_list))]
    frame_size = [config.frame_size for i in range(len(url_list))]
    (mag_spec, phase_spec) = ops.multi_processing(get_stft, zip(url_list, NFFT, frame_size), config.n_threads)

    logger.info("Extracting Spec feature succeed")

    return mag_spec, phase_spec


def mix_audio(path_1, path_2, sr=None):
    assert len(path_1) == len(path_2)
    res = []
    with open(path_1, 'r') as f:
        list_1 = f.readlines()
    with open(path_2, 'r') as f:
        list_2 = f.readlines()
    for i, j in zip(list_1, list_2):
        y1, _ = librosa.load(i, sr=sr)
        y2, _ = librosa.load(j, sr=sr)
        len_noise = len(y2)
        st_p = np.random.randint(0, len(y1) - len_noise - 1)
        res.append(np.add(y1[st_p:st_p+len_noise], y2))

