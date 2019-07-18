import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
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


def calc_eer(score_matrix, ys, save_path, plot=True, threshold_up=1.0, threshold_down=-1.0, dot_num=100000):
    if not isinstance(score_matrix, np.ndarray):
        score_matrix = np.array(score_matrix)
    if ys.shape[-1] != 1 and len(ys.shape) > 1:
        logging.warning("ys isn't 1-d array or dense index, converting.")
        ys = np.argmax(ys, -1)
    step_size = (threshold_up - threshold_down) /  (dot_num + 1)
    threshold = threshold_up
    best_eer = 1000
    if plot:  x_cord, y_cord = [], []
    for i in range(dot_num):
        threshold -= step_size
        false_negative = 0
        false_positive = 0
        for idx in range(score_matrix.shape[0]):
            for idy in range(score_matrix[idx].shape[0]):
                if score_matrix[idx][idy] < threshold and ys[idx] == idy: false_negative += 1
                if score_matrix[idx][idy] >= threshold and ys[idx] != idy: false_positive += 1
        if plot:
            x_cord.append(false_positive / (score_matrix.shape[0] * score_matrix.shape[1]))
            y_cord.append(false_negative / (score_matrix.shape[0] * score_matrix.shape[1]))
        best_eer = min(best_eer, false_negative / false_positive)
    if plot:
        figure = plt.figure()
        fig_1 = figure.add_subplot(111)
        fig_1.set_title('DET Curves')
        plt.xlabel('False Alarm probability (in%)')
        plt.ylabel('Miss Probability (in%)')
        fig_1.scatter(x_cord, y_cord, c='r', marker='.')
        plt.savefig(save_path)

