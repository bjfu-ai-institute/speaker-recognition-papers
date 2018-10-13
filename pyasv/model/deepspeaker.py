import numpy as np
from pyasv.data_manage import DataManager
from pyasv.basic import model, layers, blocks
from pyasv.loss import triplet_loss
import tensorflow as tf


class DataManage(DataManager):
    def __init__(self, config, data_type, utts_len=None, store_in_ram=False, r_data=None,
                 r_label=None, batch_num_per_file=None):
        self.utts_len = utts_len
        super().__init__(config, data_type=data_type, store_in_ram=store_in_ram,
                         r_data=r_data, r_label=r_label, batch_num_per_file=batch_num_per_file)

    def pre_process_data(self, r_data, r_label):
        self.data = []
        for i in r_data:
            while i.shape[0] < self.utts_len:  # 10s
                i = np.concatenate((i, i), 0)
            self.data.append(i[:400])
        self.data = np.array(self.data)
        self.label = np.array(r_label)

    def batch_data(self):
        bs = self.config.batch_size
        _, X, Y = self.data.shape
        self.data = self.data[:self.data.shape[0]//bs*bs].reshape(-1, bs, X, Y, 1)
        self.label = self.label[:self.label.shape[0]//bs*bs].reshape(-1, bs, 1)
        print("shape,", self.data.shape)
        print("label shape,", self.label.shape)


class DeepSpeaker(model.Model):
    def __init__(self, config, out_channel):
        self.out_channel = out_channel
        super().__init__(config)

    def inference(self, inp):
        out_channel = self.out_channel
        for i in range(len(out_channel)):
            if i > 0:
                inp_channel = inp.get_shape().as_list()[-1]
                inp = layers.conv2d(name='conv5%d'%i, shape=[5, 5, inp_channel, out_channel[i]],
                                    strides=[1, 2, 2, 1], x=inp, padding='SAME')
                inp = blocks.residual_block(inp, out_channel[i], "residual_block_%d" % i,
                                            is_first_layer=False)

            else:
                inp_channel = inp.get_shape().as_list()[-1]
                inp = layers.conv2d(name='conv5%d' % i, shape=[5, 5, inp_channel, out_channel[i]],
                                    strides=[1, 2, 2, 1], x=inp, padding='SAME')
                inp = blocks.residual_block(inp, out_channel[i], "residual_block_%d" % i,
                                            is_first_layer=True)
            print(inp.get_shape().as_list())

        inp = tf.reduce_mean(inp, axis=1)

        inp = tf.reshape(inp, [-1, inp.get_shape().as_list()[1]*inp.get_shape().as_list()[2]])
        print(inp.get_shape().as_list())
        weight_affine = layers.new_variable("affine_weight", [inp.get_shape().as_list()[-1], 512])

        bias_affine = layers.new_variable("affine_bias", [512])

        inp = tf.nn.relu(tf.matmul(inp, weight_affine) + bias_affine)
        print(inp.get_shape().as_list())
        output = tf.nn.l2_normalize(inp)
        return output

    def loss(self, embeddings, labels):
        return triplet_loss.batch_hard_triplet_loss(labels, embeddings, 0.5)
