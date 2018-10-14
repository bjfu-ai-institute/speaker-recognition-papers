import sys
sys.path.append('../..')
from pyasv.model.deepspeaker import DeepSpeaker
from pyasv.basic import ops
from pyasv.speech_processing import ext_fbank_feature
from pyasv.config import TrainConfig
from pyasv import Config
from pyasv.pipeline import TFrecordGen, TFrecordReader
import logging
import tensorflow as tf
import os
import numpy as np
import time


def no_gpu(train_data, test, enroll):
    pass


def multi_gpu(config, train_data, test=None, enroll=None):
    tf.reset_default_graph()
    logger = logging.getLogger(config.model_name)
    con = tf.ConfigProto(allow_soft_placement=True)
    con.gpu_options.allow_growth = True
    with tf.Session(config=con) as sess:
        with tf.device('/cpu:0'):
            learning_rate = config.lr
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            logger.info('build model...')
            logger.info('build model on gpu tower...')
            for gpu_id in range(config.n_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    logger.info('GPU:%d...' % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('gpu_variables', reuse=tf.AUTO_REUSE):
                            x, y = train_data.get_next()
                            model = DeepSpeaker(config, out_channel=[64, 128, 256, 512])
                            output = model.inference(x)
                            loss = model.loss(output, y)
                            grads = opt.compute_gradients(loss)
                            ops.tower_to_collection(tower_y=y, tower_losses=loss, tower_grads=grads, tower_output=output)
                        logger.info('build model on gpu tower done.')
            logger.info('reduce model on cpu...')
            aver_loss_op = tf.reduce_mean(tf.get_collection('tower_losses'))
            apply_gradient_op = opt.apply_gradients(ops.average_gradients(tf.get_collection('tower_grads')))
            all_y = tf.reshape(tf.stack(tf.get_collection('tower_y'), 0), [-1, 1])
            all_output = tf.reshape(tf.stack(tf.get_collection('tower_output'), 0), [-1, 400])
            vectors = dict()
            logger.info('reduce model on cpu done.')
            logger.info('run train op...')
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            
            for epoch in range(config.max_step):
                start_time = time.time()
                avg_loss, log_flag = 0.0, 0
                logger.info('Epoch:%d, lr:%.4f, total_batch=%d' % (epoch, config.lr, config.batch_nums_per_epoch))
                
                for batch_idx in range(config.batch_nums_per_epoch):
                    _, _loss, batch_out = sess.run([apply_gradient_op, aver_loss_op, all_output])
                    avg_loss += _loss
                    log_flag += 1
                    if log_flag % 100 == 0 and log_flag != 0:
                        log_flag = 0
                        duration = time.time() - start_time
                        start_time = time.time()
                        logger.info('At %d batch, present batch loss is %.4f, %.2f batches/sec'%(batch_idx, _loss, 100.0/duration))

                avg_loss /= config.batch_nums_per_epoch
                logger.info('Train average loss:%.4f' % (avg_loss))
                
                
                
                abs_save_path = os.path.abspath(os.path.join(config.save_path, config.model_name + ".ckpt"))
                saver.save(sess=sess, save_path=abs_save_path)

            logger.info('training done.')


def train(config, train_data, test=None, enroll=None):
    ops.system_gpu_status(config)
    if config.n_gpu == 0:
        no_gpu(train_data, test, enroll)
    else:
        multi_gpu(config, train_data, test, enroll)


def limit_len(data):
    while data.shape[0] < 100:
        data = np.concatenate((data, data), 0)
    data = data[:100]
    return data


if __name__ == '__main__':
    config = TrainConfig('../config.json')
    config.save_path = '.'
    train_urls = ['/home/data/speaker-recognition/url/train_1', '/home/data/speaker-recognition/url/train_2','/home/data/speaker-recognition/url/train_3']
    enroll_url = '/home/data/speaker-recognition/url/enroll'
    test_url = '/home/data/speaker-recognition/url/test'
    
    gen_train = TFrecordGen(config, 'Train.record')
    for train_url in train_urls:
        x, y = ext_fbank_feature(train_url, config)
        x = ops.multi_processing(limit_len, x, config.n_threads, True)
        gen_train.write(x, y)
        del x, y
    
    gen_enroll = TFrecordGen(config, 'Enroll.record')
    x, y = ext_fbank_feature(enroll_url, config)
    x = ops.multi_processing(limit_len, x, config.n_threads, True)
    gen_enroll.write(x, y)

    gen_test = TFrecordGen(config, 'Test.record')
    x, y = ext_fbank_feature(test_url, config)
    x = ops.multi_processing(limit_len, x, config.n_threads, True)
    gen_test.write(x, y)
    logger = logging.getLogger(config.model_name)

    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logger.info("Feature proccessing done.")

    train = TFrecordReader('Train.record', (100, 64), (1))
    train = train.read(config.batch_size, shuffle=True)
    train = train.prefetch(config.batch_size)
    train = train.make_one_shot_iterator()

    multi_gpu(config, train)
    """
    """
    enroll = TFrecordReader('Enroll.record', (100, 64), (1))
    enroll = enroll.read(config.batch_size, shuffle=True)
    enroll = enroll.prefetch(config.batch_size)
    
    test = TFrecordReader('Test.record', (100, 64), (1))
    test = test.read(config.batch_size, shuffle=True)
    test = test.prefetch(config.batch_size)
    """


    