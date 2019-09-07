import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import numba
from scipy.spatial.distance import cdist


colors = {'pink': '\033[95m', 'blue': '\033[94m', 'green': '\033[92m', 'yellow': '\033[93m', 'red': '\033[91m',
      'ENDC': '\033[0m', 'bold': '\033[1m', 'underline': '\033[4m'}


class AudioViewer:
    def __init__(self, save_path=None):
        self.save_path = save_path
        self.spec_ids = 0
        self.wave_ids = 0

    def draw_spec(self, y, file_name=None,x_axis='time', y_axis='linear', colorbar_format='%+2.0f dB'):
        plt.figure()
        librosa.display.specshow(y, x_axis=x_axis, y_axis=y_axis)
        plt.colorbar(format='%+2.0f dB')
        if self.save_path is None:
            plt.show()
        else:
            if file_name is None:
                plt.savefig(os.path.join(self.save_path, 'spec_%d.png'%self.spec_ids))
            else:
                plt.savefig(os.path.join(self.save_path, file_name))
            self.spec_ids += 1
        plt.close()

    def draw_wav(self, y, sr, file_name=None):
        plt.figure()
        librosa.display.waveplot(y, sr)
        if self.save_path is None:
            plt.show()
        else:
            if file_name is None:
                plt.savefig(os.path.join(self.save_path, 'wave_%d.png'%self.wave_ids))
            else:
                plt.savefig(os.path.join(self.save_path, file_name))
            self.wave_ids += 1
        plt.close()


def folder_size(path='.'):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += folder_size(entry.path)
    return total


def str_color(color, data):
    return colors[color] + str(data) + colors['ENDC']


def set_log(filename=None):
    if filename is None:
        logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(filename)s %(lineno)d] %(levelname)s: %(message)s',
                            datefmt='%m-%d %H:%M:%S')
    else:
        logging.basicConfig(filename=filename, format='%(asctime)s [%(filename)s %(lineno)d] %(levelname)s: %(message)s',
                            datefmt='%m-%d %H:%M:%S')


def get_score_matrix(embeddings, vectors, metric='cosine'):
    score_matrix = cdist(embeddings, vectors, metric=metric)
    return np.array(score_matrix)


def calc_acc(score_matrix, ys):
    if ys.shape[-1] != 1:
        label = np.argmax(ys, 1)
    else:
        label = ys
    pred = np.argmax(score_matrix, axis=1)
    Pos = np.where(label == pred)[0].shape[0]
    All = label.shape[0]
    return Pos / All


def calc_eer(score_matrix, ys, save_path, plot=True, dot_num=10000):
    if not isinstance(score_matrix, np.ndarray):
        score_matrix = np.array(score_matrix)
    if ys.shape[-1] == 1 and len(ys.shape) > 1:
        ys = ys.reshape(-1,)
        ys = np.eye(np.max(ys + 1))[ys]
        ys_con = np.ones_like(ys) - ys
    def _get_false_alarm_rate(threshold):
        pos = np.array(score_matrix >= threshold, dtype=np.int32)
        if np.sum(pos) == 0:
            return 0
        false_pos = np.array((pos - ys) > 0, dtype=np.int32)
        return np.sum(false_pos) / np.sum(pos)

    def _get_false_reject_rate(threshold):
        neg = np.array(score_matrix < threshold, dtype=np.int32)
        if np.sum(neg) == 0:
            return 0
        false_neg = np.array((neg - ys_con) > 0, dtype=np.int32)
        return np.sum(false_neg) / np.sum(neg)

    def _dichotomy(start, end, func, result, threshold=1e-5):
        mid = (start + end) / 2
        while abs(func(mid) - result) > threshold or (start - end) <= threshold:
            if func(mid) > result:
                end = mid
                mid = (mid + start) / 2
            else:
                start = mid
                mid = (mid + end) / 2
        return mid

    # threshold_up = _dichotomy(-1.0, 1.0, _get_false_reject_rate, 1e-3)
    # threshold_down = _dichotomy(-1.0, 1.0, _get_false_alarm_rate, 1e-3)
    threshold_up = 1.0
    threshold_down = -1.0

    step_size = (threshold_up - threshold_down) /  (dot_num + 1)
    threshold = threshold_up
    best_eer = 1000
    residual = 1000
    if plot:  x_cord, y_cord = [], []
    for _ in range(dot_num):
        threshold -= step_size
        fa_rate = _get_false_alarm_rate(threshold)
        fr_rate = _get_false_reject_rate(threshold)
        if plot:
            x_cord.append(fr_rate)
            y_cord.append(fa_rate)
        if residual > abs(fr_rate - fa_rate): 
            best_eer = max(fr_rate, fa_rate)
            residual = abs(fr_rate - fa_rate)
    if plot:
        figure = plt.figure()
        fig_1 = figure.add_subplot(111)
        fig_1.set_title('DET Curves')
        plt.xlabel('False Alarm probability (in%)')
        plt.ylabel('Miss Probability (in%)')
        fig_1.plot(x_cord, y_cord, c='blue')
        plt.savefig(save_path)
    return best_eer
