import functools
from typing import Optional
import scipy as sp
from scipy.fftpack import dct, idct

from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io

from functools import lru_cache

import numpy as np

from tqdm import tqdm

from common import Watermark

import matplotlib.pyplot as plt

from functools import partialmethod

from multiprocessing import Pool

class AmazonWatermark(Watermark):
    def __init__(self, n_repeats: int, n_blocks: int, beta: float, 
        high_dct: int, low_dct: Optional[int] = None, ) -> None:
        super().__init__()
        self.n_repeats = n_repeats
        self.n_blocks = n_blocks
        self.low_dct = low_dct
        self.high_dct = high_dct

        if self.low_dct is None:
            self.low_dct = 0

        self.beta = beta
        self.watermark_size = self.high_dct - self.low_dct

        assert(self.watermark_size >= self.n_blocks)

        self.signs = np.random.choice([-1, 1], size=(self.n_repeats, self.n_blocks))
        watermark = np.random.random((self.watermark_size, self.watermark_size))
        watermark = 0.5 * (watermark + watermark.T)

        _, self.watermark = np.linalg.eig(watermark)

    def encode_single(self, data: np.ndarray) -> np.ndarray:
        repeats = np.array_split(data, self.n_repeats)
        for repeat_n, repeat in enumerate(repeats):
            blocks = np.array_split(repeat, self.n_blocks)
            for block_n, block in enumerate(blocks):
                watermark_vector = self.watermark[block_n]

                dct_block = dct(block, norm="ortho", n=block.shape[0])
                center_segment = dct_block[self.low_dct:self.high_dct]

                g = np.sqrt(center_segment @ center_segment)

                center_segment = center_segment + self.beta * g * self.signs[repeat_n, block_n] * watermark_vector
                
                dct_block[self.low_dct:self.high_dct] = center_segment

                blocks[block_n] = idct(dct_block, norm="ortho")

            repeats[repeat_n] = np.concatenate(blocks)

        return np.concatenate(repeats)

    def encode(self, data: np.ndarray, watermark_length: int, watermark_step: int):
        sample_idx = 0
        data = data.copy()

        while sample_idx + watermark_length < len(data):
            print(sample_idx)
            data[sample_idx:sample_idx + watermark_length] = self.encode_single(data[sample_idx:sample_idx + watermark_length])
            sample_idx += watermark_length + watermark_step
        
        return data

    def calc_rho(self, data: np.ndarray):
        repeats = np.array_split(data, self.n_repeats)
        blocks = [np.array_split(repeat, self.n_blocks) for repeat in repeats]
        rho = 0

        y = {}

        for i in range(self.n_blocks):
            for n in range(self.n_repeats - 1):
                for m in range(n + 1, self.n_repeats):
                    if (m, i) not in y:
                        y[(m, i)] = dct(blocks[m][i], n=self.high_dct, norm="ortho")[self.low_dct:self.high_dct] * self.watermark[(i+1) % self.n_blocks]
                    if (n, i) not in y:
                        y[(n, i)] = dct(blocks[n][i], n=self.high_dct, norm="ortho")[self.low_dct:self.high_dct] * self.watermark[i]
                    y_m = y[(m, i)]
                    y_n = y[(n, i)]

                    if (np.linalg.norm(y_m) < 1) or (np.linalg.norm(y_n) < 1):
                        continue

                    h_m = np.sqrt(y_m @ y_m)
                    h_n = np.sqrt(y_n @ y_n)

                    rho += self.signs[n, i] * self.signs[m, i] * (y_n @ y_m) / h_m / h_n

        return rho

    def _calc_rho_wrapper(self, t:int, data: np.ndarray, watermark_length: int):
        return self.calc_rho(data[t:t+watermark_length])
    
    def decode(self, data: np.ndarray, watermark_length: int, sampling_length: int, 
                sampling_step: int, region_length: int, **kwargs):

        assert sampling_length < region_length

        sample_idx = 0

        results = []

        rho_cache = {}

        with Pool(4) as p:
            rho_cache = list(tqdm(p.imap(functools.partial(calc_wrapper, data=data, watermark_length=watermark_length, wmrk=self), range(data.shape[0] - watermark_length)), total=data.shape[0] - watermark_length))

        # for t in tqdm(range(data.shape[0] - watermark_length)):
        #     rho_cache[t] = self.calc_rho(data[t:t+watermark_length])


        return rho_cache


        # while sample_idx < len(data) and not np.any(results):
        #     rhos = []
        #
        #     for t in range(sample_idx, sample_idx + region_length):
        #         if t not in rho_cache:
        #             rho_cache[t] = self.calc_rho(data[t:t+watermark_length])
        #         rhos.append(rho_cache[t])
        #
        #     print(rhos)
        #
        #     rhos = np.array(rhos)
        #     rho_mean = rhos.mean()
        #     gamma = 3 * rhos.std()
        #
        #     rho = (rhos[:sampling_length] - rho_mean).mean()
        #
        #     results.append(rho >= gamma)
        #
        #     sample_idx += sampling_step
        #
        #     # print()
        #
        # return results

def calc_wrapper(t:int, data: np.ndarray, watermark_length: int, wmrk):
    return wmrk.calc_rho(data[t:t + watermark_length])
                
