import os
import tensorflow as tf
from xvector.x_vector import XVector
from pyasv.basic import ops
from pyasv.config import TrainConfig
from pyasv.pipeline import TFrecordClassBalanceReader
import time
import logging


def multi_gpu(config, train_data):
    logger = logging.getLogger(config.model_name + '_train')
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
                            x = tf.sparse_tensor_to_dense(x)
                            x = tf.reshape(x, [config.num_utt_per_class*config.num_classes_per_batch, -1, config.feature_dims])
                            y = tf.one_hot(y, depth=config.n_speaker)
                            y = tf.cast(y, dtype=tf.float32)
                            model = XVector(config)
                            output = model.inference(x)
                            tower_output.append(output)
                            loss = model.loss(y, output)
                            tower_losses.append(loss)
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
                        logger.info('build model on gpu tower done.')
            logger.info('reduce model on cpu...')
            aver_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = opt.apply_gradients(ops.average_gradients(tower_grads))
            tf.summary.scalar('loss', aver_loss_op)
            #all_output = tf.reshape(tf.stack(tower_output, 0), [-1, 512])
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
                    _, _loss, summary_str = sess.run([apply_gradient_op, aver_loss_op,  summary_op])
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

def remove_zero_vector(x):
    abs_ = tf.abs(x)
    sum_ = tf.reduce_sum(abs_, axis=-1)
    zero_vector = tf.zeros(shape=(1,1), dtype=tf.float32)
    bool_mask = tf.not_equal(sum_, zero_vector)
    omit_zeros = tf.boolean_mask(x, bool_mask)
    return omit_zeros

if __name__ == '__main__':
    config = 'x_vector.json'
    config = TrainConfig(config)
    filenames = [os.path.join(config.save_path, 'data', 'train_%d.rcd'%i) for i in range(config.n_speaker)]
    dataset = TFrecordClassBalanceReader(config, filenames)
    #with tf.Session() as sess:
        #a, b = dataset.get_next()
        #c = tf.sparse_tensor_to_dense(a)
        #print(a)
        #e = tf.reshape(c, [4, -1 ,25])
        #e = tf.RaggedTensor.from_tensor(e, padding=0) 
        #d = sess.run(e)
        ##f = sess.run(tf.map_fn(remove_zero_vector, e))
        #print(d)
        #print(d.shape)
        #print(f, f.shape)
        #print(sess.run(a))
    
    multi_gpu(config, dataset)
