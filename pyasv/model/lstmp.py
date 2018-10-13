import numpy as np
from pyasv.data_manage import DataManager


class DataManage(DataManager):
    def __init__(self, config, data_type, utts_per_batch, r_data=None, r_label=None, batch_num_per_file=None):
        super().__init__(config, data_type, r_data, r_label, batch_num_per_file)
        self.utts_num = utts_per_batch

    def pre_process_data(self, r_data, r_label):
        self.data = []
        for i in r_data:
            while i.shape[0] < 400:  # 10s
                i = np.concatenate((i, i), 0)
            self.data.append(i[:400])
        self.data = np.array(self.data)
        self.label = np.array(self.label)

    def batch_data(self):
        data = []
        label = []
        all_utts_num = []
        for i in range(self.config.N_SPEAKER):
            utts_num = len(self.data[np.where(self.label == i)])
            all_utts_num.append(utts_num)
        min_utts = np.min(all_utts_num)
        for batch in range(min_utts):
            batch_data = []
            batch_label = []
            for i in range(self.config.N_SPEAKER):
                utts = self.data[np.where(self.label == i)]
                batch_data.append(utts[:self.utts_num])
                l = np.full(shape=(self.utts_num, 1), fill_value=i)
                batch_label.append(l)
            data.append(batch_data)
            label.append(batch_label)
        self.data = np.array(data)
        self.label = np.array(label)
