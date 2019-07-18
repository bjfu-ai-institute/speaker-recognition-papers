import sys
sys.path.append('../..')
import numpy as np
import logging
from pyasv.basic import model
from pyasv import layers
from pyasv import Config
from pyasv import ops
import tensorflow as tf
import os
import time


class LSTMP(model.Model):
    def __init__(self, config, lstm_units, layer_num):
        super().__init__(config)
        self.units = lstm_units
        self._feature = None  # define in self.inference
        self.layer_num = layer_num
        self.logger = logging.getLogger('train')
        self.embed_size = lstm_units
        self._idx = [tf.fill([2], i) for i in range(config.n_speaker)]
        self.batch_size = self.config.num_classes_per_batch * self.config.num_classes_per_batch

    @property
    def feature(self):
        return self._feature

    def inference(self, x, is_training=True):
        with tf.variable_scope('Forward', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('LSTM_0', reuse=tf.AUTO_REUSE):
                outputs, state = layers.lstm(x, self.units, is_training, 1, self.config.batch_size)
            for i in range(self.layer_num-1):
                with tf.variable_scope('LSTM_%d'%(i+1), reuse=tf.AUTO_REUSE):
                    outputs, state = layers.lstm(outputs, self.units, is_training, 1, self.config.batch_size)
            #with tf.variable_scope('Output', reuse=tf.AUTO_REUSE) as scope:
            #    output = layers.full_connect(outputs[-1], name='fc_1', units=self.embed_size)
            self._feature = outputs[-1]
        return outputs[-1]

    def train(self, train_data, graph=None):
        logger = logging.getLogger('train')
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        logger.info('Build model on %s tower...'%('cpu' if self.config.n_gpu == 0 else 'gpu'))
        tower_y, tower_losses, tower_grads, tower_output = [], [], [], []
        for gpu_id in range(self.config.n_gpu):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.variable_scope('gpu_variables', reuse=tf.AUTO_REUSE):
                    x, y = train_data.get_next()
                    logging.info(x.get_shape())
                    logging.info(y.get_shape())
                    output = self.inference(x)
                    tower_output.append(output)
                    losses = self.loss(output)
                    tower_losses.append(losses)
                    grads = opt.compute_gradients(losses)
                    tower_grads.append(grads)
        aver_loss_op = tf.reduce_mean(tower_losses)
        apply_gradient_op = opt.apply_gradients(ops.average_gradients(tower_grads))
        tf.summary.scalar('loss', aver_loss_op)
        all_output = tf.reshape(tf.stack(tower_output, 0), [-1, self.embed_size])
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(self.config.save_path, 'graph'), sess.graph)
        for epoch in range(self.config.max_step):
            start_time = time.time()
            avg_loss, log_flag = 0.0, 0
            logger.info('Epoch:%d, lr:%.4f, total_batch=%d' % (epoch, self.config.lr,
                                                               self.config.batch_nums_per_epoch))
            for batch_idx in range(self.config.batch_nums_per_epoch):
                _, _loss, batch_out, summary_str = sess.run(
                    [apply_gradient_op, aver_loss_op, all_output, summary_op])
                avg_loss += _loss
                log_flag += 1
                if log_flag % 100 == 0 and log_flag != 0:
                    log_flag = 0
                    duration = time.time() - start_time
                    start_time = time.time()
                    logger.info('At %d batch, present batch loss is %.4f, %.2f batches/sec' % (
                    batch_idx, _loss, 100.0 * self.config.n_gpu / duration))
                summary_writer.add_summary(summary_str, epoch * self.config.batch_nums_per_epoch + batch_idx)
            avg_loss /= self.config.batch_nums_per_epoch
            logger.info('Train average loss:%.4f' % (avg_loss))
            abs_save_path = os.path.abspath(os.path.join(self.config.save_path,
                                                         self.config.model_name + ".ckpt"))
            saver.save(sess=sess, save_path=abs_save_path)
        logger.info('training done.')

    def predict(self, x, y, graph):
        pass

    def loss(self, embeddings, Type='softmax'):
        with tf.variable_scope("loss") as scope:
            if type(embeddings) != tf.Tensor:
                embeddings = tf.stack(embeddings)
            embeddings = tf.reshape([self.config.num_classes_per_batch,
                                     self.config.num_utt_per_class,
                                     self.embed_size])
            self.logger.debug("Embedding shape", embeddings.get_shape().as_list())
            # x - n_speaker , y - n_utt per speaker , _ - feature dims
            X, Y = self.config.num_classes_per_batch, self.config.num_utt_per_class

            b = tf.get_variable(name='GE2Eloss_b', shape=[], dtype=tf.float32)
            w = tf.get_variable(name='GE2Eloss_w', shape=[], dtype=tf.float32)

            # inp shape is [spkr_num, utt_per_spkr, embedding_dim]
            # result shape is [spkr_num, embedding_dim]
            mean_per_spkr = tf.reduce_mean(embeddings, axis=1)

            self.logger.debug("Center shape", mean_per_spkr.get_shape().as_list())

            # shape = [ spkr_num, utt_num, embedding_dim ]
            # every person's center except current utt.
            mean_except_one = (tf.reduce_sum(embeddings, axis=1, keep_dims=True) - embeddings) / (Y - 1)
            S = tf.concat(
                [tf.concat([ops.cosine(mean_except_one[i, :, :], embeddings[j, :, :]) if i == j
                            else tf.reduce_sum(mean_per_spkr[i, :] * embeddings[j, :, :], axis=1, keep_dims=True)
                            for i in range(X)], axis=1) for j in range(X)], axis=0)
            # data shape is [spkr_num, utt_per_spkr, embedding_dim]
            # The shape of S is [spkr_num, utt_per_spkr, spkr_num]
            #S = ops.cosine(embeddings, mean_per_spkr, w=w, b=b)

            self.logger.debug(S.get_shape().as_list())

            S_per_spkr = tf.reduce_sum(S, axis=-1)

            self.logger.debug(S_per_spkr.get_shape().as_list())

            L = 2 * S - S_per_spkr

            self.logger.debug(L.get_shape().as_list())
        return tf.reduce_sum(L)


if __name__ == '__main__':
    x = tf.placeholder(dtpye=tf.float32, shape=[32, 150, 64])
    config = Config('lstmp.yaml')
    model = LSTMP(config, 400, 3)
    out = model.inference(x)
    model.loss(out)