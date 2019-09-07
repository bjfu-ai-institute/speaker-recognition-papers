from pyasv.basic import ops
import tensorflow as tf
import logging


def generalized_end_to_end_loss(embeddings, w, b, loss_type='softmax', return_score_mat=False):
    """GE2E loss published by google. https://google.github.io/speaker-id/publications/GE2E/
    
    :param embeddings: `tf.float32`, Input Embedding Matrix, dim should be [n_spkr, n_utt, emb_size].
    :param w: weight of weighted cosine similarity.
    :param b: bias of weighted cosine similarity.
    :param return_score_mat: return score_mat at second, defaults to False.
    :raises AssertionError: loss type should be one of [softmax, constrast].
    :return: loss value, score_mat [n_spkr, n_utt, emb_size] (optional).
    """
    if type(embeddings) != tf.Tensor: embeddings = tf.stack(embeddings)
    # x - n_speaker , y - n_utt per speaker , _ - feature dims
    X, Y, _ = embeddings.get_shape().as_list()

    # inp shape is [spkr_num, utt_per_spkr, embedding_dim]
    # result shape is [spkr_num, embedding_dim]
    mean_per_spkr = ops.normalize(tf.reduce_mean(embeddings, axis=1))
    
    # shape = [ spkr_num, utt_num, embedding_dim ]
    # every person's center except current utt.
    mean_except_one = ops.normalize((tf.reduce_sum(embeddings, axis=1, keepdims=True) - embeddings) / (Y - 1))
    _s = tf.squeeze(tf.abs(w) * tf.stack([[ops.cosine(mean_except_one[i, :, :], embeddings[j, :, :]) if i == j
                        else ops.cosine(mean_per_spkr[i, :], embeddings[j, :, :])
                        for i in range(X)] for j in range(X)]) + b, axis=-1)
    _s = tf.transpose(_s, [0, 2, 1])
    
    # shape of S [spkr_num, spkr_num, utt_num]
    # S[j][i][k] mean j_th people's i_th audio embedding's score to k_th people

    if loss_type == 'softmax':
        # get i-th people's all audios' score to himself.
        _s_correct = tf.stack([_s[i, :, i] for i in range(X)])

        # false score minus true score.
        # sum of all minus the resut of true score times 2.
        _l = - 2 * _s_correct + tf.log(tf.reduce_sum(tf.exp(_s), axis=-1) + 1e-10)
    elif loss_type == 'contrast':
        _s_per_spkr = tf.reduce_max(_s, axis=-1)
        _l = tf.stack([1 - tf.sigmoid(_s[i * X:(i + 1) * X, :, i]) + _s_per_spkr[i, :] for i in range(X)])
    else:
        raise AssertionError("loss_type should be one of [`softmax`, `contrast`]")
    if return_score_mat: return tf.reduce_sum(_l), _s
    else: return tf.reduce_sum(_l)
