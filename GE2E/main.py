import sys
import os
import logging
import h5py
import argparse
import numpy as np
import time
import collections
import tensorflow as tf

def initialize():
    # get config.
    config = pyasv.Config(config=FLAGS.conf_path)
    return config


def read_data(config):
    """Feature store in "project_folder/data" read them as 'tf.data.dataset' object.

    :param config: Object contain hyper parameters' setting.
    :return: dataset object for training or testing.
    """
    data_path = os.path.join(config.save_path, 'data')

    keys_to_feature = collections.OrderedDict([('data', tf.FixedLenFeature(shape=[(config.fix_len * config.sample_rate) // config.hop_length,
                                                                                   config.feature_dims],
                                                                           dtype=tf.float32)),
                                               ('label', tf.FixedLenFeature(shape=[1], dtype=tf.float32))])

    if FLAGS.is_training:
        filenames = set([os.path.join(data_path, i) if i[:5] == 'train' else '' for i in os.listdir(data_path)])
        if '' in filenames:  filenames.remove('')

        data = pyasv.TFrecordClassBalanceReader(config, list(filenames), keys_to_feature)
        enroll = h5py.File(os.path.join(config.save_path, "data/enroll.h5"), 'r')
        test = h5py.File(os.path.join(config.save_path, "data/test.h5"), 'r')
        valid = {'e_x': enroll['data'], 'e_y': np.reshape(enroll['label'], (-1, 1)),
                 't_x': test['data'], 't_y': np.reshape(test['label'], (-1, 1))}
        return data, valid

    elif FLAGS.is_testing:
        enroll = h5py.File(os.path.join(config.save_path, "data/enroll.h5"), 'r')
        test = h5py.File(os.path.join(config.save_path, "data/test.h5"), 'r')
        pred_data = {'e_x': enroll['data'], 'e_y': np.reshape(enroll['label'], (-1, 1)),
                     't_x': test['data'], 't_y': np.reshape(test['label'], (-1, 1))}
        return pred_data


def handle_path(config):
    backup_name = time.strftime("backup-%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
    for dd in ['log', 'graph', 'model']:
        if os.path.exists(os.path.join(config.save_path, dd)):
            try:
                os.mkdir(os.path.join(config.save_path, backup_name))
            except:
                pass
            os.rename(os.path.join(config.save_path, dd),
                      os.path.join(config.save_path, backup_name, dd))
            logging.info('Moving %s to backup' % dd)
        os.mkdir(os.path.join(config.save_path, dd))


def prepare_wav_to_id(url_path, config):
    # Get all "$wav_path $spkr" files.
    urls = os.listdir(FLAGS.data_url)
    train, enroll, test = [], None, None
    for url in urls:
        if url[:5] == 'train':
            train.append(os.path.join(FLAGS.data_url, url))
        elif url[:6] == 'enroll':
            enroll = os.path.join(FLAGS.data_url, url)
        elif url[:4] == 'test':
            test = os.path.join(FLAGS.data_url, url)

    # Backup old log, url or model, won't backup if is_restore is true.
    if FLAGS.is_training and not FLAGS.is_restore:
        if os.path.exists(url_path) and pyasv.utils.folder_size(url_path) > 0 and not FLAGS.is_restore:
            logging.warning("save_path/url is exist and not empty, using old url file.")
        else:
            if FLAGS.is_restore:
                logging.warning("Can't find url file or log file, but is_restore is true.")
            if not os.path.exists(url_path):
                os.mkdir(url_path)

            # change "$wav_path $spkr" => "$wav_path $id" and save to url_path as *_tmp_*
            # and we can get speaker number and reset the default number to Preventing default number error.
            n_speaker, n_speaker_test = create_record.create_url(config, urls=train, enroll=enroll, test=test)
            config.set_value(n_speaker=n_speaker, n_speaker_test=n_speaker_test)
            config.save('lstmp.yaml')



def extract_feature(data_path, config):
    if os.path.exists(data_path) and pyasv.utils.folder_size(data_path) > 0:
        logging.warning("save_path/data is exist and not empty, assume feature have been extracted.")
    else:
        if FLAGS.is_restore:
            logging.warning("Can't find data, but is_restore is true.")
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        url_folder = os.path.join(config.save_path, 'url')

        # extract enroll test feature to h5 file.
        enroll_ext = pyasv.speech.FilterBank(url_folder=url_folder, config=config, file_name='enroll')
        train, enroll, test = enroll_ext.read_url_file()
        enroll_ext.extract_h5(enroll)

        # extract train feature to class balance tfrecord.
        if FLAGS.is_training:
            train_ext = pyasv.speech.FilterBank(url_folder=url_folder, config=config, file_name='train')
            train_ext.extract_class_balance_rcd(train)
            test_ext = pyasv.speech.FilterBank(url_folder=url_folder, config=config, file_name='test')
            test_ext.extract_h5(test)

        # extract test feature to h5.
        if FLAGS.is_testing:
            test_ext = pyasv.speech.FilterBank(url_folder=url_folder, config=config, file_name='test')
            test_ext.extract_h5(test)


def run():
    """Running project

    :param train: list of train scp file path.
    :param enroll: str of enroll scp file path.
    :param test: str of test scp file path.
    """

    config = initialize()
    data_path = os.path.join(config.save_path, 'data')
    url_path = os.path.join(config.save_path, 'url')

    # mkdir or backup
    handle_path(config)

    # set log
    config.set_project_loggers()

    # prepare metadata files
    prepare_wav_to_id(url_path, config)

    extract_feature(data_path, config)

    if FLAGS.is_training:
        data, valid = read_data(config)
        model = lstmp.LSTMP(config, FLAGS.units, FLAGS.layer, FLAGS.prob)
        model.train(data, valid)

    if FLAGS.is_testing:
        data = read_data(config)
        model = lstmp.LSTMP(config, FLAGS.units, FLAGS.layer)
        model.predict(data, FLAGS.model_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data_url', help="Url folder of data",
                        default='/home/fhq/data/aishell/url/aishell-small')
    parser.add_argument('--tool', dest='tool_path', help='Path to pyasv project.',
                        default='../')
    parser.add_argument('--conf', dest='conf_path', help='Path to config file.',
                        default='lstmp.yaml')
    parser.add_argument('--model_dir', dest='model_dir', help="Path of trained model",
                        type=str, default="None")
    parser.add_argument('--prob', default=-1, dest="prob", help="Probability of dropout",
                        type=float, )
    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--is_testing', action='store_true')
    parser.add_argument('--units', dest='units', default=400)
    parser.add_argument('--layer', dest='layer', default=3)
    parser.add_argument('--emb', dest='embedding_size', default=512)
    parser.add_argument('--is_restore', action='store_true')
    FLAGS = parser.parse_args()

    sys.path.append(FLAGS.tool_path)

    import pyasv
    import create_record, lstmp

    pyasv.utils.set_log()

    run()
