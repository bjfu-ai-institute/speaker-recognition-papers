import numpy
import scipy
import h5py
import sys
sys.path.append('../..')
import pyasv


class PLDA:
    def __init__(self, load_path=None, config=None, data=None):
        # reading parameters.
        if load_path is not None:
            self.load(load_path)
        else:
            tmp_x, tmp_y = data.next_batch
            self.vector_size = tmp_x.shape[1]
            self.rank_f = config.PLDA_F_RANK
            self.rank_g = config.PLDA_G_RANK
            self.num_vector = data.num_examples
            self.max_step = config.MAX_STEP
            self.class_num = config.N_SPEAKER
            self.name = config.MODEL_NAME
            self.save_path = config.SAVE_PATH
            data.reset_batch_counter()
            # initial mean and residual covariance.
            self.mean = numpy.mean(data.raw_frames, axis=0)
            self.sigma = self._initial_sigma(data.raw_frames)

            # initial EigenVoice matrix
            self.F = self._initial_F()
            self.G = self._initial_G()
            # convert data of per speaker.
            data = self._data_per_speaker(data)

            # Optimize parameters by EM algorithm
            self._EM_loop(data)

    def load(self, path):
        with h5py.File(path, 'r') as f:
            self.F = f['f']
            self.G = f['g']
            self.sigma = f['sigma']
            self.mean = f['mean']

    def write(self, name):
        with h5py.File(name, 'w') as f:
            f.create_dataset('mean', data=self.mean, compression='gzip')
            f.create_dataset('f', data=self.F, compression='gzip')
            f.create_dataset('sigma', data=self.sigma, compression='gzip')
            f.create_dataset('G', data=self.G, compression='gzip')
            print("write %s succeed"%name)

    def score(self, enroll, test):
        # center data.
        spkr_num = enroll.spkr_num
        labels = numpy.array(enroll.raw_labels, dtype=numpy.int32)
        enroll = numpy.array(enroll.raw_frames, dtype=numpy.float32)
        test = numpy.array(test.raw_frames, dtype=numpy.float32)
        enroll_ = numpy.zeros([spkr_num, self.vector_size], dtype=numpy.float32)
        for i in range(spkr_num):
            enroll_[i] = numpy.mean(enroll[numpy.argmax(labels, 1) == i], 0)
        enroll = enroll_

        enroll -= self.mean
        test -= self.mean

        # Compute temporary matrices
        inv_sigma = numpy.linalg.inv(self.sigma)
        I_iv = numpy.eye(self.mean.shape[0], dtype=numpy.float32)
        I_ch = numpy.eye(self.G.shape[1], dtype=numpy.float32)
        I_spk = numpy.eye(self.F.shape[1], dtype=numpy.float32)
        A = numpy.linalg.inv(self.G.T.dot(inv_sigma).dot(self.G) + I_ch)
        B = self.F.T.dot(inv_sigma).dot(I_iv - self.G.dot(A).dot(self.G.T).dot(inv_sigma))
        K = B.dot(self.F)
        K1 = numpy.linalg.inv(K + I_spk)
        K2 = numpy.linalg.inv(2 * K + I_spk)

        # Compute Gaussian distribution constant
        alpha1 = numpy.linalg.slogdet(K1)[1]
        alpha2 = numpy.linalg.slogdet(K2)[1]
        constant = alpha2 / 2.0 - alpha1

        # Compute score
        test_tmp = B.dot(test.T)
        enroll_tmp = B.dot(enroll.T)
        tmp1 = test_tmp.T.dot(K1)

        S1 = numpy.empty(test.shape[0])
        for i in range(test.shape[0]):
            S1[i] = tmp1[i, :].dot(test_tmp[:, i])/2
        S2 = numpy.empty(enroll.shape[0])
        S = numpy.empty([enroll.shape[0], test.shape[0]])
        for i in range(enroll.shape[0]):
            mod_p_test_seg = test_tmp + numpy.atleast_2d(enroll_tmp[:, i]).T
            tmp2 = mod_p_test_seg.T.dot(K2)
            S2[i] = enroll_tmp[:, i].dot(K1).dot(enroll_tmp[:, i])/2.0
            S[i, :] = numpy.einsum("ij, ji->i", tmp2, mod_p_test_seg)/2.0

        S += constant - (S1 + S2[:, numpy.newaxis])
        with open("./log", 'w') as f:
            for i in S:
                f.writelines(str(i) + '\n')
        print(S.shape)
        return S

    def _data_per_speaker(self, data):
        data_dic = numpy.zeros([self.class_num, self.vector_size], dtype=numpy.float32)
        for i in range(self.class_num):
            data_dic[i] = data.raw_frames[numpy.argmax(data.raw_labels, axis=1) == i].sum(axis=0)
        return data_dic

    def _whiten(self, data, sigma, mu=None):
        eigen_values, eigen_vectors = scipy.linalg.eigh(sigma)
        ind = eigen_values.real.argsort()[::-1]
        eigen_values = eigen_values.real[ind]
        eigen_vectors = eigen_vectors.real[:, ind]

        eigen_values = eigen_values.real
        eigen_values[eigen_values < 0] = 1.2e-2
        sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values)
        sqr_inv_sigma = numpy.dot(eigen_vectors, numpy.diag(sqr_inv_eval_sigma))

        # center the data.
        if mu is not None:
            data -= mu
        data = numpy.dot(data, sqr_inv_sigma)
        return data

    def _initial_G(self):
        return numpy.random.randn(self.vector_size, self.rank_g)

    def _initial_sigma(self, feature):
        C = feature - numpy.mean(feature, axis=0)
        return numpy.dot(C.T, C) / self.vector_size

    def _initial_F(self):
        evals, evecs = scipy.linalg.eigh(self.sigma)
        idx = numpy.argsort(evals)[::-1]
        evecs = evecs.real[:, idx[:self.rank_f]]
        return evecs[:, :self.rank_f]

    def _EM_loop(self, data):
        print("\n-------------")
        for step in range(self.max_step):
            print("Estimate between class covariance, step=%d, max_step=%d"%(step+1, self.max_step))

            print("E-step....")
            tmp_data = data
            # whiten the data and EigenVoice matrix
            tmp_data = self._whiten(tmp_data, self.sigma, self.mean)
            eigen_values, eigen_vectors = scipy.linalg.eigh(self.sigma)
            ind = eigen_values.real.argsort()[::-1]
            eigen_values = eigen_values.real[ind]
            eigen_vectors = eigen_vectors.real[:, ind]
            eigen_values = eigen_values.real
            eigen_values[eigen_values < 0] = 1.2e-2
            sqr_inv_eval_sigma = 1 / numpy.sqrt(eigen_values)
            sqr_inv_sigma = numpy.dot(eigen_vectors, numpy.diag(sqr_inv_eval_sigma))

            self.F = sqr_inv_sigma.T.dot(self.F)
            self.G = sqr_inv_sigma.T.dot(self.G)

            e_h = numpy.zeros([self.class_num, self.rank_f])
            e_hh = numpy.zeros([self.class_num, self.rank_f, self.rank_f])
            A = numpy.dot(self.F.T, self.F)
            inv_lambda = scipy.linalg.inv(A + numpy.eye(A.shape[0]))
            for batch in range(self.class_num):
                aux = numpy.dot(self.F.T, data[batch, :])
                e_h[batch] = numpy.dot(aux, inv_lambda)
                e_hh[batch] = inv_lambda + numpy.outer(e_h[batch], e_h[batch])

            _R = numpy.sum(e_hh, axis=0) / self.class_num
            _C = e_h.T.dot(tmp_data).dot(scipy.linalg.inv(sqr_inv_sigma))
            _A = numpy.einsum('ijk,i->jk', e_hh, numpy.ones(self.class_num))

            print("M-step...")
            self.F = scipy.linalg.solve(_A, _C).T
            self.sigma -= numpy.dot(self.F, _C) / self.num_vector
            self.F = numpy.dot(self.F, scipy.linalg.cholesky(_R))
            self.G = self._maximization(self.G, _A, _C, _R)

            if self.name is None:
                self.name = 'plda'
            else:
                self.name = self.name
            name = '%s-it%d.h5'%(self.name, step+1)
            if self.save_path is None:
                self.save_path = './'
            self.write(self.save_path + name)

    def _maximization(self, phi, _A, _C, _R):
        d = self.vector_size
        ch = scipy.linalg.cholesky(_R)
        phi = numpy.linalg.solve(_A, _C[:, 0:d]).T
        phi = numpy.dot(phi, ch)
        return phi


if __name__ == '__main__':
    pass