import matplotlib.pyplot as plt
import numpy as np

from common import Watermark
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pywt
import math

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class DWTEnsembleWatermark(Watermark):
    def __init__(self, watermark_length: int, beta: float):
        self.watermark_length = watermark_length
        self.train_size = 2000
        self.k = self.train_size + watermark_length
        self.beta = beta
        self.watermark = np.random.choice([0, 1], (self.watermark_length, ))
        self.train_seq = np.random.choice([0, 1], (self.train_size, ))
        self.full_seq = np.concatenate((self.train_seq, self.watermark))

        self.wavelet_type = 'haar'

    def encode(self, data: np.ndarray, **kwargs):
        original_length = data.shape[0]
        l = math.ceil(data.shape[0] / (self.k * 2))
        data = np.pad(data, (0, self.k * 2 * l - data.shape[0]))
        frames = np.split(data, self.k)

        subframes = [(frame[::2], frame[1::2]) for frame in frames]
        wavelets = [(pywt.wavedec(s[0], self.wavelet_type, level=2), pywt.wavedec(s[1], self.wavelet_type, level=2)) for s in subframes]
        coeffs = np.array([(fst[0], snd[0]) for fst, snd in wavelets])
        energies = np.sum(np.square(coeffs), axis=-1) / coeffs.shape[0]
        new_energies = np.zeros(energies.shape)
        print(np.mean(energies, axis=0))

        cnt = 0

        for i, bit in enumerate(self.full_seq):
            e1, e2 = energies[i]
            de = abs(e1 - e2) + self.beta

            if bit == 0:
                if e1 > e2:
                    if self.beta < abs(e1 - e2):
                        e1 = e1 - de / 2
                        e2 = e2 + de / 2
                    # else:
                    #     cnt += 1
            elif bit == 1:
                if e1 <= e2:
                    if self.beta < abs(e1 - e2):
                        e1 = e1 + de / 2
                        e2 = e2 - de / 2
                    # else:
                    #     cnt += 1

            new_energies[i] = (e1, e2)

        print(cnt / self.k)


        dum_preds = new_energies[:, 0] > new_energies[:, 1]

        print(accuracy_score(self.full_seq, dum_preds))

        coeffs = np.sign(coeffs) * np.sqrt(np.square(coeffs) * np.expand_dims(new_energies, axis=-1) / np.expand_dims(energies + 1e-4, axis=-1))
        new_wavelets = [([fst] + wavelets[i][0][1:], [snd] + wavelets[i][1][1:]) for i, (fst, snd) in enumerate(coeffs)]
        new_subframes = [(pywt.waverec(fst, self.wavelet_type), pywt.waverec(snd, self.wavelet_type)) for (fst, snd) in new_wavelets]

        new_subframes_1, new_subframes_2 = list(zip(*new_subframes))

        for i in range(len(frames)):
            frames[i][::2] = new_subframes_1[i][:l]
            frames[i][1::2] = new_subframes_2[i][:l]

        new_data = np.zeros(data.shape)

        for i in range(len(frames)):
            new_data[i*2*l:(i+1)*2*l] = frames[i]

        return new_data[:original_length]

    def decode(self, data: np.ndarray, **kwargs):
        l = math.ceil(data.shape[0] / (self.k * 2))
        data = np.pad(data, (0, self.k * 2 * l - data.shape[0]))
        frames = np.split(data, self.k)

        subframes = [(frame[::2], frame[1::2]) for frame in frames]
        wavelets = [(pywt.wavedec(s[0], self.wavelet_type, level=2), pywt.wavedec(s[1], self.wavelet_type, level=2)) for s in subframes]
        coeffs = np.array([(fst[0], snd[0]) for fst, snd in wavelets])
        energies = np.sum(np.square(coeffs), axis=-1) / coeffs.shape[0]

        plt.plot(energies[:, 0] - energies[:, 1])
        # plt.plot(energies[:, 1])


        deltas = (energies[:, 0] - energies[:, 1]).reshape((-1, 1))

        pred_data = np.hstack((deltas, energies[:, 0].reshape((-1, 1))))

        train_X = pred_data[:self.train_size]
        train_y = self.train_seq

        test_X = pred_data[self.train_size:]
        test_y = self.watermark

        clf = RandomForestClassifier()
        clf.fit(train_X, train_y)

        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(train_X, train_y)

        clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf2.fit(train_X, train_y)

        # vclf = VotingClassifier(estimators=[
        #     ('rf', clf), ], voting = 'soft')
        #
        # vclf.fit(train_X, train_y)


        print(accuracy_score(test_y, clf.predict(test_X)))
        print(accuracy_score(test_y, neigh.predict(test_X)))
        print(accuracy_score(test_y, clf2.predict(test_X)))

        # print(accuracy_score(test_y, vclf.predict(test_X)))


        dum_preds = energies[:, 0] > energies[:, 1]
        print(accuracy_score(self.full_seq, dum_preds))

        print()



