import numpy as np
import sys
sys.path.append('..')
import os
import config

class DataManage(object):
    def __init__(self, raw_frames, raw_labels, batch_size):
        assert len(raw_frames) == len(raw_labels)
        # must be one-hot encoding
        self.raw_frames = np.array(raw_frames, dtype=np.float32)
        self.raw_labels = np.array(raw_labels, dtype=np.float32)
        self.batch_size = batch_size
        self.epoch_size = len(raw_frames) / batch_size
        self.batch_counter = 0
        self.spkr_num = np.array(self.raw_labels).shape[-1]

    @property
    def next_batch(self):
        if (self.batch_counter+1) * self.batch_size < len(self.raw_frames):
            batch_frames = self.raw_frames[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size]
            batch_labels = self.raw_labels[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size]
            self.batch_counter += 1
            return batch_frames, batch_labels
        else:
            return self.raw_frames[self.batch_counter * self.batch_size:-1], \
            self.raw_labels[self.batch_counter * self.batch_size:-1]

class DataManage4BigData(object):
    """
    Use this for huge dataset
    """
    def __init__(self, url):
        self.url = url
        self.batch_count = 0
        if os.path.exists(self.url) and os.listdir(self.url):
            self.file_is_exist = True
        else:
            self.file_is_exist = False

    def write_file(self, raw_frames, raw_labels):
        batch_size = config.BATCH_SIZE
        local_batch_count = 0
        if type(raw_frames) == np.ndarray:
            data_length = raw_frames.shape[0]
        else:
            data_length = len(raw_frames)
        while local_batch_count * batch_size < data_length:
            if local_batch_count * (batch_size+1) >= data_length:
                frames = raw_frames[local_batch_count * batch_size:]
                labels = raw_labels[local_batch_count * batch_size:]
                np.savez_compressed(os.path.join(self.url, "data_%d.npz"%local_batch_count), frames=frames, labels=labels)    
            else:
                frames = raw_frames[local_batch_count * batch_size: (local_batch_count+1) * batch_size]
                labels = raw_labels[local_batch_count * batch_size: (local_batch_count+1) * batch_size]
                np.savez_compressed(os.path.join(self.url, "data_%d.npz"%local_batch_count), frames=frames, labels=labels)    
                local_batch_count += 1

    @property
    def next_batch(self):
        if not self.file_is_exist:
            print('You need write file before load it.')
            exit()
        loaded = np.load(os.path.join(self.url, "data_%d.npz"%self.batch_count))
        frames = loaded['frames']
        labels = loaded['labels']
        self.batch_count += 1
        return frames, labels
        