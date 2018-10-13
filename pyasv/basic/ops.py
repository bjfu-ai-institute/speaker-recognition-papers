import tensorflow as tf
import numpy as np
import os
from scipy.spatial.distance import cosine
import multiprocessing as mp
import sys


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y):
    for i in range(len(models)):
        x, y, _, _, _ = models[i]
        start_pos = i * payload_per_gpu
        stop_pos = (i + 1) * payload_per_gpu
        inp_dict[x] = batch_x[start_pos:stop_pos]
        inp_dict[y] = batch_y[start_pos:stop_pos]
    return inp_dict


def update_embeddings(vectors, embeddings, ys, config):
    for spkr in range(config.n_speaker):
        if embeddings[np.argmax(ys, 1) == spkr]:
            vector = np.mean(embeddings[np.where(np.argmax(ys, 1) == spkr)], axis=0)
            if spkr in vectors.keys():
                vector = (vectors[spkr] + vector) / 2
            else:
                vector = vector
            vectors[spkr] = vector
        else:
            if spkr not in vectors.keys():
                vectors[spkr] = np.zeros(512, dtype=np.float32)
    return vectors


def system_gpu_status(config):
    if config.n_gpu == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    else:
        if os.path.exists('./tmp'):
            os.rename('./tmp', './tmp-backup')
        if sys.platform[:3] == 'win':
            os.system("powershell \"nvidia-smi -q -d Memory | Select-String Free > ./tmp\"")
            memory_gpu = open('tmp', 'r', encoding='utf-16').readlines()[1:-2]
            memory_gpu = [int(x.split()[2]) for x in memory_gpu]
            mem_ = []
            for i in range(len(memory_gpu)):
                if i%2 == 0:
                    mem_.append(memory_gpu[i])
            memory_gpu = mem_
        else:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
            memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        memory_gpu = np.array(memory_gpu, dtype=np.int32)
        gpu_list = []
        for gpu in range(config.n_gpu):
            gpu_list.append(str(np.argmax(memory_gpu)))
            memory_gpu[np.argmax(memory_gpu)] = -10000
        s = ""
        for i in range(config.n_gpu):
            if i != 0:
                s += ','
            s += str(gpu_list[i])
        os.environ['CUDA_VISIBLE_DEVICES'] = s
        os.remove('./tmp')


def tower_to_collection(**kwargs):
    for key in kwargs.keys():
        tf.add_to_collection('key', kwargs[key])


def get_score_matrix(embeddings, vectors):
    score_matrix = []
    for utt in embeddings:
        row = []
        for spkr in vectors.keys():
            score = cosine(vectors[spkr], utt)
            row.append(score)
        score_matrix.append(row)
    return np.array(score_matrix)


def calc_acc(score_matrix, ys):
    if ys.shape[-1] != 1:
        label = np.argmax(ys, 1)
    else:
        label = ys
    pred = np.argmax(score_matrix, axis=1)
    Pos = np.where(label == pred)[0].shape[0]
    All = label.shape[0]
    return Pos / All


def calc_eer(score_matrix, ys):
    pass


def multi_processing(func, jobs, proccess_num, use_list_params=False):
    with mp.Pool(proccess_num) as pool:
        if not use_list_params:
            res = pool.starmap(func, jobs)
        else:
            res = pool.map(func, jobs)
    return res
