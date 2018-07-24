"""
ext_mfcc_feature
----------------
.. autofunction:: pyasv.speech_processing.ext_mfcc_feature


ext_fbank_feature
-----------------

.. autofunction:: pyasv.speech_processing.ext_fbank_feature


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
-------------

.. autofuncion: pyasv.speech_processing.cmvn
"""
import librosa
import os
import numpy as np
from scipy import signal
import scipy
import resampy
from scipy.io.wavfile import read
from scipy.fftpack import dct


def slide_windows(feature):
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
    for i in range(feature.shape[0]-49):
        if i < 50:
            continue
        else:
            result_.append(np.array(feature[i-50:i+50]))
    result = np.array(result_)  
    return result
        

def ext_mfcc_feature(url_path):
    """This function is used for extract MFCC feature of a dataset.

    Parameters
    ----------
    url_path : ``str``
        The path of the 'PATH' file.

    Notes
    -----
    The file contain the path to all audio and its number in the dataset.
    The contents of the file should be as follows:

    xxxxx/your_data_path/1_1.wav 0

    xxxxx/your_data_path/1_2.wav 0

    xxxxx/your_data_path/2_1.wav 1

    Returns
    -------
    fbank : ``list``
        The feature array. each frame concat with the frame before and after it.
    label : ``list``
        The label of fbank feature.

    ::
        #TODO：changeable concat size.
    """
    with open(url_path, 'r') as urls:
        mfccs = []
        labels = []
        for url in list(urls):
            url, label = str(url).split(" ")
            index = eval(str(label).split("\n")[0])
            y, sr = librosa.load(url)
            mfcc_ = librosa.feature.mfcc(y, sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc_, width=3)
            mfcc_delta_delta = librosa.feature.delta(mfcc_delta, width=3)
            mfcc = np.vstack([mfcc_, np.vstack([mfcc_delta, mfcc_delta_delta])])
            mfcc = cmvn(mfcc)
            mfcc = slide_windows(mfcc)
            for i in mfcc:
                mfccs.append(i)
                labels.append(index)
        return mfccs, labels


def ext_fbank_feature(url_path):
    """This function is used for extract features of one dataset.

    Parameters
    ----------
    url_path : ``str``
        The path of the 'PATH' file.

    Notes
    -----
    The file contain the path to all audio and its number in the dataset.
    The contents of the file should be as follows:

    xxxxx/your_data_path/1_1.wav 0

    xxxxx/your_data_path/1_2.wav 0

    xxxxx/your_data_path/2_1.wav 1

    Returns
    -------
    fbank : ``list``
        The feature array. each frame concat with the frame before and after it.
    label : ``list``
        The label of fbank feature.

    Todo
    ----
    changeable concat size.

    """
    with open(url_path, 'r') as urls:
        fbanks = []
        labels = []
        for url in list(urls):
            url, label = str(url).split(" ")
            index = eval(str(label).split("\n")[0])
            #label = np.zeros(config.N_SPEAKER)
            #label[index] = 1
            """
            # I am not sure I use librosa in right way.
            y, sr = librosa.load(url)
            stft = librosa.core.stft(y)
            melW=librosa.filters.mel(sr=sr, n_fft=2048,n_mels=40,fmin=0.,fmax=22100)
            melW /= np.max(melW, axis=-1)[:,None]
            fbank = np.dot(stft.T, melW.T)
            fbank = slide_windows(fbank)
            """
            fbank = calc_fbank(url)
            fbank = slide_windows(fbank)
            for i in fbank:
                fbanks.append(i)
                labels.append(index)
        return fbanks, labels


def calc_fbank(url):
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
    pre_emphasis = 0.97
    frame_size = 0.025
    frame_stride = 0.010
    NFFT = 512
    nfilt = 64
    
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    # convert from seconds to samples
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))

    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(emphasized_signal, z) 

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    filter_banks = cmvn(filter_banks)
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
    fs_orig: ``int``
        origin sample rate
    fs_new: ``int``
        new sample rate
    axis: ``int``
        the resample axis

    Returns
    -------
    spec_res : spectrogram after resample
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
    ``np.ndarray``
        the feature after cmvn.
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
