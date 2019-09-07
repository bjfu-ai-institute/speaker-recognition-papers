import sys
sys.path.append('../..')
from lstmp import LSTMP
from pyasv.basic import ops
from pyasv.config import Config
import logging
import tensorflow as tf
import time


def no_gpu(train_data, test, enroll):
    pass


def multi_gpu(config, train_data):
    logger = logging.getLogger(config.model_name)
    con = tf.ConfigProto(allow_soft_placement=True)
    con.gpu_options.allow_growth = True
    with tf.Session(config=con) as sess:
        with tf.device('/cpu:0'):
            learning_rate = config.lr
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            logger.info('build model...')
            tower_y, tower_losses, tower_grads, tower_output = [], [], [], []
            for gpu_id in range(config.n_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('gpu_variables', reuse=tf.AUTO_REUSE):
                            x = train_data.get_next()
                            model = LSTMP(config, 512, 3)
                            output = model.train_inference(x)
                            tower_output.append(output)
                            loss = model.loss(output)
                            tower_losses.append(loss)
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
            aver_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = opt.apply_gradients(ops.average_gradients(tower_grads))

            tf.summary.scalar('loss', aver_loss_op)
            all_output = tf.reshape(tf.stack(tower_output, 0), [-1, 512])
            vectors = dict()
            logger.info('reduce model on cpu done.')
            logger.info('run train op...')
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('log', sess.graph)
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
                abs_save_path = os.path.abspath(os.path.join(config.save_path, config.model_name + ".ckpt"))
                saver.save(sess=sess, save_path=abs_save_path)

            logger.info('training done.')


def parse(proto):
    global data_shape
    keys_to_features = {
        'data': tf.FixedLenFeature(shape=data_shape, dtype=tf.float32),
        'label': tf.FixedLenFeature(shape=(1), dtype=tf.int64),
    }
    parsed_ = tf.parse_single_example(proto, keys_to_features)
    return parsed_['data']


if __name__ == '__main__':
    config = '../pyasv/config.json'
    config = Config(config)
    #reader = TFrecordReader('../pyasv/model/Train.record', (100, 64), (1))
    num_classes = 800
    num_classes_per_batch = 200
    num_utt_per_class = 2
    import os
    filenames = [os.getcwd() + '/save/' + 'Train%d.record'%i for i in range(num_classes)]

    datasets = [tf.data.TFRecordDataset(f).map(parse, num_parallel_calls=16).repeat(None) for f in filenames]


    def generator(_):
        # Sample `num_classes_per_batch` classes for the batch
        sampled = tf.random_shuffle(tf.range(num_classes))[:num_classes_per_batch]
        # Repeat each element `num_images_per_class` times
        batch_labels = tf.tile(tf.expand_dims(sampled, -1), [1, num_utt_per_class])
        return tf.to_int64(tf.reshape(batch_labels, [-1]))


    selector = tf.contrib.data.Counter().map(generator)
    selector = selector.apply(tf.contrib.data.unbatch())

    dataset = tf.contrib.data.choose_from_datasets(datasets, selector)

    # Batch
    batch_size = num_classes_per_batch * num_utt_per_class
    dataset = dataset.batch(batch_size)

    dataset = dataset.make_one_shot_iterator()

    multi_gpu(config, dataset)
