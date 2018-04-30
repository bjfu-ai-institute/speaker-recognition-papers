import numpy as np


class DataManage(object):
    def __init__(self, raw_frames, raw_labels, batch_size,
                 enrollment_frames=None, enrollment_targets=None):
        assert len(raw_frames) == len(raw_labels)
        # must be one-hot encoding
        self.raw_frames = np.array(raw_frames, dtype=np.float32)
        self.raw_labels = np.array(raw_labels, dtype=np.float32)
        self.batch_size = batch_size
        self.epoch_size = len(raw_frames) / batch_size
        self.enrollment_frames = np.array(enrollment_frames, dtype=np.float32)
        self.enrollment_targets = np.array(enrollment_targets, dtype=np.float32)
        self.batch_counter = 0
        self.enroll_vector = []
        self.spkr_num = np.array(self.raw_labels).shape[-1]

    def next_batch(self):
        if (self.batch_counter+1) * self.batch_size < len(self.raw_frames):
            batch_frames = self.raw_frames[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size+1]
            batch_labels = self.raw_labels[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size+1]
            self.batch_counter += 1
            return batch_frames, batch_labels
        else:
            return self.raw_frames[self.batch_counter * self.batch_size:-1], \
            self.raw_labels[self.batch_counter * self.batch_size:-1]

    @property
    def pred_data(self):
        return self.enrollment_frames, self.enrollment_targets, self.raw_frames, self.raw_labels
