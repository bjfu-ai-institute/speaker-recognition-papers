import numpy as np
from pyasv.data_manage import DataManager
from pyasv.basic import model
from pyasv.basic import layers
import tensorflow as tf


class DataManage(DataManager):
    def __init__(self, config, data_type, utts_per_batch, store_in_ram=False,
                 r_data=None, r_label=None, batch_num_per_file=None):
        self.utts_num = utts_per_batch
        super().__init__(config, data_type, store_in_ram=store_in_ram,
                         r_data=r_data, r_label=r_label,
                         batch_num_per_file=batch_num_per_file)

    def pre_process_data(self, r_data, r_label):
        self.data = []
        for i in r_data:
            while i.shape[0] < 400:  # 10s
                i = np.concatenate((i, i), 0)
            self.data.append(i[:400])
        self.data = np.array(self.data)
        self.label = np.array(r_label)

    def batch_data(self):
        data = []
        label = []
        all_utts_num = []
        for i in range(self.config.n_speaker):
            utts_num = len(self.data[np.where(self.label == i)])
            if utts_num != 0:
                all_utts_num.append(utts_num)
        min_utts = np.min(all_utts_num)
        print(min_utts)
        for batch in range(min_utts // self.utts_num):
            print(min_utts // self.utts_num)
            batch_data = []
            batch_label = []
            for i in range(self.config.n_speaker):
                utts = self.data[np.where(self.label == i)]
                batch_data.append(utts[batch*self.utts_num:(batch+1)*self.utts_num])
                l = np.full(shape=(self.utts_num, 1), fill_value=i)
                batch_label.append(l)
            data.append(batch_data)
            label.append(batch_label)
        self.data = np.array(data)
        self.label = np.array(label)


class LSTMP(model.Model):
    def __init__(self, config, LSTM_units, layer_num):
        super().__init__(config)
        self.units = LSTM_units
        self._feature = None
        self.layer_num = layer_num

    def inference(self, x, is_training=True):
        outputs, state = layers.lstm(x, self.units, is_training, self.layer_num)
        output = layers.full_connect(state, name='fc_1', units=400)
        self._feature = output
        return output

    @property
    def feature(self):
        return self._feature

    def loss(self, embeddings, Type='softmax'):
        if type(embeddings) != tf.Tensor:
            embeddings = tf.stack(embeddings)

        # x - n_speaker , y - n_utt per speaker , _ - feature dims
        X, Y, _ = embeddings.get_shape().as_list()

        b = tf.get_variable(name='GE2Eloss_b', shape=[])
        w = tf.get_variable(name='GE2Eloss_w', shape=[])
        S = []  # similar matrix
        L = []
        for i in range(X):                   # speaker index
            spk_list = []
            for j in range(Y):               # utterance index
                radial_diffs = tf.multiply(
                    tf.gather(tf.gather(embeddings, i), j),
                    tf.reduce_mean(tf.gather(embeddings, j), axis=1))
                score = w*(1 - tf.reduce_sum(radial_diffs, keepdims=True)) + b
                spk_list.append(score)
            spk_loss = []
            for j in range(Y):
                loss = S[j][i] - tf.log(tf.exp(tf.reduce_sum(spk_list)))
                spk_loss.append(loss)
            S.append(spk_list)
            L.append(spk_loss)
        # S shape is [N_speaker, N_utts]
        return tf.reduce_sum(L)
