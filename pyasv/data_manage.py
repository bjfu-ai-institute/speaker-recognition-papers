"""
DataManage
----------

.. autoclass:: DataManage
    :members:

    .. automethod:: __init__

DataManage4BigData
------------------

.. autoclass:: DataManage4BigData
    :members:

    .. automethod:: __init__
"""
import numpy as np
import os
import h5py
import shutil


class DataManager(object):
    def __init__(self, config, data_type, store_in_ram=False, r_data=None, r_label=None, batch_num_per_file=None):
        self.batch_num_per_file = batch_num_per_file
        self.config = config
        self.read_batch_counter = 0
        self.url = os.path.join(self.config.SAVE_PATH, 'data', data_type)
        self.store_in_ram = store_in_ram
        if r_data is None and r_label is None:
            info = h5py.File(os.path.join(self.url, 'info.h5'), 'a')
            self.batch_nums = info['counter'].value
        else:
            self.data = None
            self.label = None
            self.pre_process_data(r_data, r_label)
            self.batch_data()
            if not store_in_ram:
                self.write_file()

    def pre_process_data(self, r_data, r_label):
        try:
            self.data = np.array(r_data, dtype=np.float32)
        except ValueError:
            print("Audio length is not equal.")
            exit()
        self.label = np.array(r_label)

    def batch_data(self):
        _, l, d = self.data.shape
        self.data = self.data.reshape(-1, self.config.BATCH_SIZE, l, d)
        if len(self.label.shape) == 1:
            self.label = self.label.reshape(-1, self.config.BATCH_SIZE, 1)
        else:
            d = self.label.shape[1]
            self.label = self.label.reshape(-1, self.config.BATCH_SIZE, d)

    def _write_batch(self, data, label, batch_st):
        data_file = h5py.File(os.path.join(self.url,
                                           'batch_%d-%d.h5') % (batch_st, batch_st+self.batch_num_per_file-1))
        data_file.create_dataset(name='data', data=data, compression='gzip')
        data_file.create_dataset(name='label', label=label, compression='gzip')
        data_file.close()

    def write_file(self):
        info_url = os.path.join(self.url, 'info.h5')
        info = h5py.File(info_url, 'a')
        if 'counter' not in info.keys():
            counter = 0
        else:
            counter = info['counter'].value
        if 'tmp_data' in info.keys():
            self.data = np.concatenate((info['tmp_data'].value, self.data), 0)
            del info['tmp_data']

        utts_num = self.data.shape[0]
        res_num = utts_num%self.config.BATCH_SIZE
        file_num = utts_num//self.batch_num_per_file
        tmp_data = self.data[-res_num:]
        self.data = np.split(self.data[:-res_num], file_num)
        tmp_label = self.label[:-res_num]

        if file_num < 0:
            print("Warning: Passed a dataset which shape is smaller than batch_num_per_file, will save to info.h5 'tmp_data'.")


        for batch in range(file_num):
            print("batch start: ", batch + counter)
            self._tmp_data_file = h5py.File(os.path.join(self.url, 'batch_%d-%d.h5') % ((batch+counter),
                                          (batch+counter+self.batch_num_per_file-1)), 'a')
            data.create_dataset(name='data',
                                data=self.data[batch*self.batch_num_per_file:(batch+1)*self.batch_num_per_file],
                                compression='gzip')
            data.create_dataset(name='label',
                                data=self.label[batch*self.batch_num_per_file:(batch+1)*self.batch_num_per_file],
                                compression='gzip')
            data.close()
            counter += self.batch_num_per_file
        try:
            del info['counter']
        except KeyError:
            pass


        if utts_num % self.batch_num_per_file != 0:
            info.create_dataset(name='tmp_data',
                                data=self.data[file_num*self.batch_num_per_file:],
                                compression='gzip')
        del self.data
        del self.label
        info.create_dataset(name='counter', data=counter)
        info.close()

    def next_batch(self):
        if self.store_in_ram:
            data = self.data[self.read_batch_counter]
            label = self.label[self.read_batch_counter]
            self.read_batch_counter += 1
            return data, label
        else:
            bc = self.read_batch_counter
            if bc % self.batch_num_per_file == 0:
                st = (bc // self.batch_num_per_file)*self.batch_num_per_file
                data = h5py.File(os.path.join(self.url, 'batch_%d-%d.h5'%(st, st+self.batch_num_per_file-1)), 'a')
                self.data = data['data'].value
                self.label = data['label'].value
                self.counter4file = 0
            ret_data = self.data[self.counter4file]
            ret_label = self.label[self.counter4file]
            self.counter4file += 1
            return ret_data, ret_label

    def reset_batch_counter(self):
        self.read_batch_counter = 0




class DataManage(object):
    """
    Use ``DataManage`` to manage normal dataset,
    we use next_batch to get batch data every step.
    """
    def __init__(self, raw_frames, raw_labels, config):
        """
        Parameters
        ----------
        raw_frames : ``list`` or ``np.ndarray``
            the feature array of a dataset.
        raw_labels : ``list`` or ``np.ndarray``
            the label array of a dataset.
        config : ``config`` class
            The config of your model, we only need use its 'batch_size' member.
        """
        assert len(raw_frames) == len(raw_labels)
        # must be one-hot encoding
        raw_frames, raw_labels = self.random_shuffle_union(raw_frames, raw_labels)
        self.raw_frames = np.array(raw_frames, dtype=np.float32)
        raw_labels = np.array(raw_labels)
        if raw_labels.shape[-1] != config.N_SPEAKER:
            raw_labels = np.eye(config.N_SPEAKER)[raw_labels.reshape(-1)]
        self.raw_labels = np.array(raw_labels, dtype=np.float32)
        self.batch_size = config.BATCH_SIZE
        if type(raw_frames) == np.ndarray:
            self.num_examples = raw_frames.shape[0]
        else:
            self.num_examples = len(raw_frames)
        self.epoch_size = self.num_examples / config.BATCH_SIZE
        self.batch_counter = 0
        self.spkr_num = np.array(self.raw_labels).shape[-1]

    def reset_batch_counter(self):
        self.batch_counter = 0

    @property
    def next_batch(self):
        """``property`` to get next batch data.

        Returns
        -------
        batch_frames : ``list`` or ``np.ndarray``.
        batch_labels : ``list`` or ``np.ndarray``.

        Notes
        -----
        the type of ``batch_frames`` and ``batch_labels`` is
        same as your input data.
        """
        if (self.batch_counter+1) * self.batch_size <= len(self.raw_frames):
            batch_frames = self.raw_frames[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size]
            batch_labels = self.raw_labels[self.batch_counter * self.batch_size:
                                           (self.batch_counter+1) * self.batch_size]
            self.batch_counter += 1
            return batch_frames, batch_labels
        else:
            return self.raw_frames[self.batch_counter * self.batch_size:], \
                   self.raw_labels[self.batch_counter * self.batch_size:]

    @staticmethod
    def random_shuffle_union(x, y):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        return x, y


class DataManage4BigData(object):
    """
    Use ``DataManage4BigData`` to manage huge dataset.
    we will save the dataset to disk, and restore a batch of them
    in each step we can still use next_batch to get batch data
    every step.
    """
    def __init__(self, config, split_type):
        """
        Parameters
        ----------
        config : ``config`` class
            the config of your model. we will use its batch_size to manage our data
            and save the data to save_path/data.
        """
        self.batch_size = config.BATCH_SIZE
        self.split_type = split_type
        self.batch_count = 0
        self.path = config.SAVE_PATH
        if not os.path.exists(os.path.join(self.path, 'data')):
            os.mkdir(os.path.join(self.path, 'data'))
        self.url = os.path.join(self.path, 'data', split_type)

        if not os.path.exists(self.url):
            os.mkdir(self.url)
        self.spkr_num = config.N_SPEAKER
        if not os.path.exists(os.path.join(self.url, 'write_count.h5')):
            self.write_count = 0
            self.num_examples = 0
        else:
            with h5py.File(os.path.join(self.url, 'write_count.h5')) as f:
                self.write_count = f['write_count'].value
                self.num_examples = self.write_count * self.batch_size
        if os.path.exists(self.url) and os.listdir(self.url):
            self.file_is_exist = True
        else:
            self.file_is_exist = False

    def reset_batch_counter(self):
        self.batch_count = 0

    def write_file(self, raw_frames, raw_labels):
        """Save your data to save_path/data.

        Parameters
        ----------
        raw_frames : ``list`` or ``np.ndarray``
            the feature array of your dataset.
        raw_labels : ``list`` or ``np.ndarray``
            the label array of your dataset.
        """
        batch_size = self.batch_size
        local_batch_count = 0

        raw_frames, raw_labels = self.random_shuffle_union(raw_frames, raw_labels)

        raw_labels = np.array(raw_labels)
        if raw_labels.shape[-1] != self.spkr_num:
            raw_labels = np.eye(self.spkr_num)[raw_labels.reshape(-1)]
        if type(raw_frames) == np.ndarray:
            data_length = raw_frames.shape[0]
            self.num_examples = data_length
        else:
            data_length = raw_frames.shape[0]
            self.num_examples = data_length
            print("Total number of batches to be written to disk: ", int(data_length//batch_size))
        while 1:
            if batch_size * (local_batch_count+1) > data_length:
                break
            else:
                frames = raw_frames[local_batch_count * batch_size: (local_batch_count+1) * batch_size]
                labels = raw_labels[local_batch_count * batch_size: (local_batch_count+1) * batch_size]
                batch_length = ((local_batch_count+1) * batch_size) - (local_batch_count * batch_size)
                print("Writing data/%s to disk : Batch "%self.split_type +
                      str(self.write_count)+" having length "+str(batch_length))
                np.savez_compressed(os.path.join(self.url, "data_%d.npz" % self.write_count),
                                    frames=frames, labels=labels)
                self.write_count += 1
                self.num_examples = self.write_count * self.batch_size
                local_batch_count += 1
        self.file_is_exist = True
        if not os.path.exists(os.path.join(self.url, 'write_count.h5')):
            with h5py.File(os.path.join(self.url, 'write_count.h5')) as f:
                f.create_dataset('write_count', data=self.write_count)
        else:
            with h5py.File(os.path.join(self.url, 'write_count.h5')) as f:
                del f['write_count']
                f.create_dataset('write_count', data=self.write_count)

    def clear(self):
        shutil.rmtree(self.url)
        self.write_count = 0
        self.num_examples = 0
        self.batch_count = 0
        self.file_is_exist = False

    @property
    def next_batch(self):
        """Read a batch of data.

        Returns
        -------
        frames : ``np.ndarray``
            the feature array of your dataset.
        labels : ``np.ndarray``
            the label array of your dataset.

        Notes
        -----
        we will check if the data is saved. If you didn't save the data
        you will get log and two empty array.

        """
        if not self.file_is_exist:
            print('You need write file before load it.')
            return np.array([]), np.array([])
        if self.batch_count > self.write_count:
            print("Warning: batch count is over batch number, auto reset batch count.")
            self.reset_batch_counter()
        loaded = np.load(os.path.join(self.url, "data_%d.npz"%self.batch_count))
        frames = loaded['frames']
        labels = loaded['labels']
        self.batch_count += 1
        return frames, labels

    @staticmethod
    def random_shuffle_union(x, y):
        if type(x) != np.ndarray or type(y) != np.ndarray:
            x = np.array(x)
            y = np.array(y)
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        return x, y
