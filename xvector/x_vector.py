import sys
sys.path.append('../..')
import numpy as np
from pyasv.basic import model
from pyasv.basic import layers
from pyasv.basic import ops
from pyasv.basic import utils
from scipy.spatial.distance import cdist
import time
from pyasv.config import Config
from pyasv import loss
import logging
import tensorflow as tf
import os


class XVector(model.Model):
    def __init__(self, config):
        super().__init__(config)
        self.embed_size = 512
        self.n_speaker = config.n_speaker
        self.n_speaker_test = config.n_speaker_test

    def inference(self, x, is_training=True):
        with tf.variable_scope('Forward', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('T_DNN_1', reuse=tf.AUTO_REUSE):
                out_1 = layers.t_dnn(x, out=512, length=2, strides=1, name='',
                             init=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('T_DNN_2', reuse=tf.AUTO_REUSE):
                out_2 = layers.t_dnn(out_1, length=2, strides=1, name='',
                                     init=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('T_DNN_3', reuse=tf.AUTO_REUSE):
                out_3 = layers.t_dnn(out_2, length=3, strides=1, name='',
                                     init=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('T_DNN_4', reuse=tf.AUTO_REUSE):
                out_4 = layers.t_dnn(out_3, length=1, strides=1, name='',
                                     init=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('T_DNN_5', reuse=tf.AUTO_REUSE):
                out_5 = layers.t_dnn(out_4, length=1, strides=1, name='',
                                     init=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope('Pooling', reuse=tf.AUTO_REUSE):
                out_6 = layers.static_pooling(out_5)
            with tf.variable_scope('FC_1', reuse=tf.AUTO_REUSE):
                out_7 = layers.full_connect(out_6, name="", units=512)
            with tf.variable_scope('FC_2', reuse=tf.AUTO_REUSE):
                out_8 = layers.full_connect(out_7, name="", units=512)
            if is_training:
                with tf.variable_scope('Out', reuse=tf.AUTO_REUSE):
                    out_9 = layers.full_connect(out_8, name="", units=self.config.n_speaker)
                return out_9, out_8
            else:
                return out_8

    def loss(self, y_, y):
        y = tf.one_hot(tf.cast(y, dtype=tf.int64), depth=self.n_speaker)
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y)

    def ge2e_loss(self, embeddings, loss_type='softmax'):
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            b = tf.get_variable(name='loss_b', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(-5.0))
            w = tf.get_variable(name='loss_w', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(10.0))
            embeddings = tf.reshape(embeddings, [self.config.num_classes_per_batch,
                                                 self.config.num_utt_per_class,
                                                 self.embed_size])
            return loss.generalized_end_to_end_loss(embeddings, w=w, b=b)

    def train(self, train_data, valid=None):
        """Interface to train model.

        :param train_data: `tf.data.dataset`
        :param valid: dict, defaults to None. contain enroll and test data,
                            like {'t_x:0': [...], 'e_x:0': [...], 'e_y:0': ...}
        """
        logger = logging.getLogger('train')
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        logger.info('Build model on %s tower...' % ('cpu' if self.config.n_gpu == 0 else 'gpu'))
        tower_y, tower_losses, tower_grads, tower_output = [], [], [], []
        for gpu_id in range(self.config.n_gpu):
            with tf.device('/gpu:%d' % gpu_id):
                x, y = train_data.get_next()
                x = tf.reshape(x, [-1, 251, 64])
                y = tf.reshape(y, [-1, 1])
                output, vector = self.inference(x)
                tower_output.append(output)
                losses = self.loss(output, y)
                tower_losses.append(losses)
                grads = ops.clip_grad(opt.compute_gradients(losses), 3.0)
                grads = [(0.01 * i, j) if (j.name == 'loss/loss_b:0' or j.name == 'loss/loss_w:0') else (i, j) for i, j
                         in grads]
                tower_grads.append(grads)
        # handle batch loss
        aver_loss_op = tf.reduce_mean(tower_losses)
        apply_gradient_op = opt.apply_gradients(ops.average_gradients(tower_grads))
        tf.summary.scalar('loss', aver_loss_op)

        # init
        emb = self.init_validation()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(self.config.save_path, 'graph'), sess.graph)
        log_flag = 0

        for epoch in range(self.config.max_step):
            logger.info('Epoch:%d, lr:%.4f, total_batch=%d' %
                        (epoch, self.config.lr, self.config.batch_nums_per_epoch))
            avg_loss = 0.0
            start_time = time.time()
            for batch_idx in range(self.config.batch_nums_per_epoch):
                _, _loss, summary_str = sess.run([apply_gradient_op, aver_loss_op, summary_op])
                avg_loss += _loss
                log_flag += 1
                if log_flag % 100 == 0 and log_flag != 0:
                    duration = time.time() - start_time
                    start_time = time.time()
                    logger.info('At %d batch, present batch loss is %.4f, %.2f batches/sec' %
                                (batch_idx, _loss, 100 * self.config.n_gpu / duration))
                if log_flag % 600 == 0 and log_flag != 0:
                    test_x, test_y, enroll_x, enroll_y = valid['t_x'], valid['t_y'], valid['e_x'], valid['e_y']
                    acc, _ = self._validation(emb, test_x, test_y, enroll_x, enroll_y, sess, step=epoch)
                    logger.info('At %d epoch after %d batch, acc is %.6f'
                                % (epoch, batch_idx, acc))
                summary_writer.add_summary(summary_str, epoch * self.config.batch_nums_per_epoch + batch_idx)
            avg_loss /= self.config.batch_nums_per_epoch
            logger.info('Train average loss:%.4f' % avg_loss)
            abs_save_path = os.path.abspath(os.path.join(self.config.save_path, 'model',
                                                         self.config.model_name + ".ckpt"))
            saver.save(sess=sess, save_path=abs_save_path)
        logger.info('training done.')

    def _validation(self, emb, test_x, test_y, enroll_x, enroll_y, sess, step=0):
        t_emb = sess.run(emb, feed_dict={"t_x:0": test_x})
        e_emb = sess.run(emb, feed_dict={"t_x:0": enroll_x})
        # emb_file = h5py.File(os.path.join(self.config.save_path, 'log', 'mean_embeddings_%d.h5')%step, 'w')
        spkr_embeddings = np.array([np.mean(e_emb[enroll_y.reshape(-1, ) == i], 0)
                                    for i in range(self.n_speaker_test)], dtype=np.float32)
        # emb_file.create_dataset(name='enroll', data=spkr_embeddings)
        score_mat = np.array([np.reshape(1 - cdist(spkr_embeddings[i].reshape(1, self.embed_size), t_emb, metric='cosine'), (-1,))
                              for i in range(self.n_speaker_test)]).T
        print(score_mat)
        score_idx = np.argmax(score_mat, -1)
        # emb_file.create_dataset(name='score_mat', data=score_mat)
        # emb_file.close()
        return np.sum(score_idx == test_y.reshape(-1, )) / score_idx.shape[0], score_mat

    def init_validation(self):
        """Get validation operation."""
        inp = tf.placeholder(dtype=tf.float32, shape=[None, 251, self.config.feature_dims], name='t_x')
        # score_mat = self._valid(p_test_x, p_enroll_x, p_enroll_y)
        emb = self.inference(inp, is_training=False)
        return emb

    def predict(self, data, model_dir):
        with tf.Session() as sess:
            logger = logging.getLogger('train')
            emb = self.init_validation()
            saver = tf.train.Saver()
            saver.restore(sess, model_dir)
            test_x, test_y, enroll_x, enroll_y = data['t_x'], data['t_y'], data['e_x'], data['e_y']
            acc, score_mat = self._validation(emb, test_x, test_y, enroll_x, enroll_y, sess)
            eer = utils.calc_eer(score_mat, test_y,
                                 save_path=os.path.join(self.config.save_path, 'graph', 'eer.png'))
            logger.info("acc: %.6f \teer: %.6f" % (acc, eer))
            return acc, eer