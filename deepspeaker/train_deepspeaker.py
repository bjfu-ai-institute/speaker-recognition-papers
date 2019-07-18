from deepspeaker.deepspeaker import DeepSpeaker
from pyasv.basic import ops
from pyasv.speech_processing import ext_fbank_feature
from pyasv.config import TrainConfig
from pyasv.pipeline import TFrecordGen, TFrecordReader
import logging
import tensorflow as tf
import os
import numpy as np
import time


def no_gpu(train_data, test, enroll):
    pass


def multi_gpu(config, train_data):
    logger = logging.getLogger(config.model_name+'_train')
    con = tf.ConfigProto(allow_soft_placement=True)
    con.gpu_options.allow_growth = True
    with tf.Session(config=con) as sess:
        with tf.device('/cpu:0'):
            learning_rate = config.lr
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            logger.info('build model...')
            logger.info('build model on gpu tower...')

            tower_y, tower_losses, tower_grads, tower_output = [], [], [], []

            for gpu_id in range(config.n_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    logger.info('GPU:%d...' % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('gpu_variables', reuse=tf.AUTO_REUSE):
                            x, y = train_data.get_next()
                            model = DeepSpeaker(config, out_channel=[64, 128, 256, 512])
                            tower_y.append(y)
                            output = model.inference(x)
                            tower_output.append(output)
                            loss = model.loss(output, y)
                            tower_losses.append(loss)
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
                        logger.info('build model on gpu tower done.')
            logger.info('reduce model on cpu...')
            aver_loss_op = tf.reduce_mean(tower_losses)

            
            apply_gradient_op = opt.apply_gradients(ops.average_gradients(tower_grads))

            tf.summary.scalar('loss', aver_loss_op)
            all_y = tf.reshape(tf.stack(tower_y, 0), [-1, 1])
            all_output = tf.reshape(tf.stack(tower_output, 0), [-1, 512])
            vectors = dict()
            logger.info('reduce model on cpu done.')
            logger.info('run train op...')
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(config.save_path + '/log/graph', sess.graph)
            for epoch in range(config.max_step):
                start_time = time.time()
                avg_loss, log_flag = 0.0, 0
                logger.info('Epoch:%d, lr:%.4f, total_batch=%d' % (epoch, config.lr, config.batch_nums_per_epoch))
                
                for batch_idx in range(config.batch_nums_per_epoch):
                    _, _loss, batch_out, summary_str = sess.run([apply_gradient_op, aver_loss_op, all_output, summary_op])
                    avg_loss += _loss
                    log_flag += 1
                    if log_flag % 100 == 0 and log_flag != 0:
                        log_flag = 0
                        duration = time.time() - start_time
                        start_time = time.time()
                        logger.info('At %d batch, present batch loss is %.4f, %.2f batches/sec'%(batch_idx, _loss, 100.0*config.n_gpu/duration))
                    summary_writer.add_summary(summary_str, epoch*config.batch_nums_per_epoch+batch_idx)

                avg_loss /= config.batch_nums_per_epoch
                logger.info('Train average loss:%.4f' % (avg_loss))
                
                abs_save_path = os.path.abspath(os.path.join(config.save_path+'/model', config.model_name + ".ckpt"))
                saver.save(sess=sess, save_path=abs_save_path)

            logger.info('training done.')

def limit_len(data):
    while data.shape[0] < 100:
        data = np.concatenate((data, data), 0)
    data = data[:100]
    return data


def restore(config, enroll, test, test_num):
    with tf.Session() as sess:
        x, y = enroll.get_next()
        sess.run(enroll.initializer)
        sess.run(test.initializer)
        if config.n_gpu == 0:
            model = DeepSpeaker(config, out_channel=[64, 128, 256, 512])
        else:
            with tf.variable_scope('gpu_variables', reuse=tf.AUTO_REUSE):
                model = DeepSpeaker(config, out_channel=[64, 128, 256, 512])
        vec = model.inference(x)
        saver = tf.train.Saver()
        abs_save_path = os.path.abspath(os.path.join(config.save_path, config.model_name  + ".ckpt"))
        saver.restore(sess, abs_save_path)
        x, y = enroll.get_next()

        print("restore model succeed.")

        vec, y_ = sess.run([vec, y])
        spkr_dict = {}
        ops.update_embeddings(spkr_dict, vec, y_, config)
        x, y =test.get_next()
        new_vec, ans = [], []
        for i in range(test_num):
            b_vec, b_ans = sess.run([vec, y])
            new_vec.append(b_vec)
            ans.append(b_ans)
        new_vec = np.concatenate(new_vec, 0)
        ans = np.concatenate(ans, 0)
        score_matrix = ops.get_score_matrix(new_vec, spkr_dict)

        with open('result.txt') as f:
            for i in score_matrix:
                f.writelines(i)
        acc = ops.calc_acc(score_matrix, ans)
        print(acc)


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
    logger = logging.getLogger(config.model_name+'_train')

    #logger.info("Feature proccessing done.")

    train = TFrecordReader('Train.record', (100, 64), (1))
    train = train.read(config.batch_size, shuffle=True)
    train = train.prefetch(config.batch_size)
    train.repeat(None)
    train = train.make_one_shot_iterator()
    """

    #multi_gpu(config, train)
    enroll = TFrecordReader('Enroll.record', (100, 64, 1), (1))
    enroll = enroll.read(1420, shuffle=True)
    enroll = enroll.prefetch(1420)
    enroll = enroll.make_one_shot_iterator()
    test = TFrecordReader('Test.record', (100, 64, 1), (1))
    test = test.read(100, shuffle=True)
    test = test.prefetch(100)
    test = test.make_one_shot_iterator()
    restore(config, enroll, test, test_num=500)

    """
    ops.system_gpu_status(config)
    data_shape=(100,64,1)
    #config = '../config.json'
    #config = TrainConfig(config)
    #reader = TFrecordReader('../pyasv/model/Train.record', (100, 64), (1))
    
    num_classes = 1400
    num_classes_per_batch = 4
    num_utt_per_class = 16
    filenames = ['/fast/fhq' + '/save/' + 'Train%d.record'%i for i in range(num_classes)]
    datasets = [tf.data.TFRecordDataset(f).map(tf.data.TFRecordDataset.parse, num_parallel_calls=16).repeat(None)\
                for f in filenames]

    def generator(_):
        # Sample `num_classes_per_batch` classes for the batch
        sampled = tf.random_shuffle(tf.range(num_classes))[:num_classes_per_batch]
        # Repeat each element `num_images_per_class` times
        batch_labels = tf.tile(tf.expand_dims(sampled, -1), [1, num_utt_per_class])
        return tf.to_int64(tf.reshape(batch_labels, [-1]))

    selector = tf.contrib.data.Counter().map(generator)
    y = selector.batch(1)
    y_it = y.make_one_shot_iterator()
    selector = selector.apply(tf.contrib.data.unbatch())

    dataset = tf.contrib.data.choose_from_datasets(datasets, selector)
    #selector = selector.batch(batch_size)

    # Batch
    batch_size = num_classes_per_batch * num_utt_per_class
    dataset = dataset.batch(64)

    dataset = dataset.make_one_shot_iterator()

    multi_gpu(config, dataset, y_it)

