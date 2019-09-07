from pyasv.basic import model, layers, blocks
from pyasv.loss import triplet_loss
import tensorflow as tf


class DeepSpeaker(model.Model):
    def __init__(self, config, out_channel):
        self.out_channel = out_channel
        super().__init__(config)

    def inference(self, inp):
        out_channel = self.out_channel
        for i in range(len(out_channel)):
            with tf.name_scope("res_%d"%i):
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

        inp = tf.reduce_mean(inp, axis=1)

        inp = tf.reshape(inp, [-1, inp.get_shape().as_list()[1]*inp.get_shape().as_list()[2]])
        #print(inp.get_shape().as_list())
        weight_affine = layers.new_variable("affine_weight", [inp.get_shape().as_list()[-1], 512])

        bias_affine = layers.new_variable("affine_bias", [512])

        inp = tf.matmul(inp, weight_affine) + bias_affine
        #print(inp.get_shape().as_list())
        output = tf.nn.l2_normalize(inp)
        return output

    def loss(self, embeddings, labels):
        return triplet_loss.batch_hard_triplet_loss(labels, embeddings, 0.5)

    def softmax_loss(self, embeddings, labels, config):
        y_ = layers.full_connect(embeddings, name='softmax_loss', units=config.n_speaker,activation=None)
        y = tf.one_hot(tf.reshape(labels, (-1,)), depth=config.n_speaker)
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)

    def centerloss(self, embeddings, labels, alpha, config):
        dims = embeddings.get_shape()[-1]
        n_class = config.n_speaker
        centers = tf.get_variable('centers', [n_class, dims], dtype=tf.float32, initializer=tf.constant_initializer(0),
                                  trainable=False)
        labels = tf.reshape(labels, [-1])

        center_mat = tf.gather(centers, labels)

        loss = tf.nn.l2_loss(embeddings - center_mat)

        diff = center_mat - embeddings

        unique_label, unique_idx, unique_count= tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff = diff / tf.cast(1 + appear_times, tf.float32)
        diff = alpha * diff

        return loss, diff


