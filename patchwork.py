import functools
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from common import Watermark

import math

from scipy.fftpack import dct, idct
import math

import numpy as np
from scipy.fftpack import dct, idct

from common import Watermark


class PatchworkWatermark(Watermark):
    def __init__(self, m: int, alpha: int = 12, delta: int = 0.8):
        assert (m % 3 == 0)
        self.m = m
        self.alpha = alpha
        self.delta = delta
        self.sync_code = np.random.choice([0, 1], size=(m // 3,))
        self.watermark = np.random.choice([0, 1], size=(m // 3,))
        self.full_seq = np.concatenate((self.sync_code, self.sync_code, self.watermark))

    def calc_fdlm(self, coeffs: np.ndarray):
        tmp = np.abs(coeffs) / self.alpha
        # print(tmp)
        return np.abs(np.sum(np.log2(tmp, out=np.zeros_like(tmp), where=(tmp != 0)))) / coeffs.shape[0]

    def encode(self, data: np.ndarray, n: int, **kwargs):
        n1 = math.floor(data.shape[0] / n / self.m)
        if n1 % 2 == 1:
            n1 -= 1
        working_length = n1 * n * self.m
        frames = np.split(data[:working_length], n)

        for frame_n in range(n):
            segments = np.split(frames[frame_n], self.m)

            k = int(n1 // 2 * 3 / 4)

            coeffs = np.array(
                [[dct(segment[:n1 // 2], norm="ortho"), dct(segment[n1 // 2:], norm="ortho")] for segment in
                 segments])

            k_coeffs = coeffs[:, :, :k]

            fdlm = np.apply_along_axis(self.calc_fdlm, -1, k_coeffs)
            # qfdlm = fdlm
            qfdlm = np.int32(np.floor(fdlm / self.delta)) * self.delta + self.delta / 2
            rqfdlm = np.abs(np.int32(np.floor(fdlm[:, 0] / self.delta)) - np.int32(np.floor(fdlm[:, 1] / self.delta)))

            wmrk_mask = rqfdlm % 2 == self.full_seq

            qfdlm[wmrk_mask, 0] = qfdlm[wmrk_mask, 0] + self.delta // 2
            qfdlm[np.logical_not(wmrk_mask), 1] = qfdlm[np.logical_not(wmrk_mask), 1] - self.delta // 2

            scaling_coeff = np.expand_dims(np.divide(qfdlm, fdlm, out=np.zeros(qfdlm.shape), where=(fdlm != 0)),
                                           axis=-1)

            k_coeffs = np.sign(k_coeffs) * np.power(np.abs(k_coeffs), scaling_coeff * np.power(self.alpha, 1 - scaling_coeff))
            coeffs[:, :, :k] = k_coeffs
            inv_coeffs = np.apply_along_axis(lambda x: idct(x, norm="ortho"), -1, coeffs)
            new_segments = np.concatenate((inv_coeffs[:, 0, :], inv_coeffs[:, 1, :]), axis=-1)

            frames[frame_n] = np.concatenate(new_segments)

        new_data = np.zeros_like(data)
        new_data[:working_length] = np.concatenate(frames)
        new_data[working_length:] = data[working_length:]
        return new_data

    def decode_single(self, frame: np.ndarray):
        segments = np.split(frame, self.m)
        n1 = segments[0].shape[0]

        k = int(n1 // 2 * 3 / 4)

        coeffs = np.array(
            [[dct(segment[:n1 // 2], norm="ortho"), dct(segment[n1 // 2:], norm="ortho")] for segment in
             segments])

        k_coeffs = coeffs[:, :, :k]

        fdlm = np.apply_along_axis(self.calc_fdlm, -1, k_coeffs)
        rqfdlm = np.abs(np.int32(np.floor(fdlm[:, 0] / self.delta)) - np.int32(np.floor(fdlm[:, 1] / self.delta)))

        decoded_bits = np.int32(rqfdlm / self.delta) % 2

        return [np.logical_xor(decoded_bits[:self.m // 3], decoded_bits[self.m // 3: 2*self.m // 3]).sum(), np.logical_xor(decoded_bits[-self.m // 3:], self.watermark).sum()]

    def decode(self, data: np.ndarray, n: int, **kwargs):
        n1 = math.floor(data.shape[0] / n / self.m)
        if n1 % 2 == 1:
            n1 -= 1
        frame_size = self.m * n1

        with Pool(4) as p:
            sync_values = list(tqdm(
                p.imap(functools.partial(decode_wrapper, data=data, frame_size=frame_size, wmrk=self),
                       range(data.shape[0] - frame_size)), total=data.shape[0] - frame_size))

        return [x for x in sync_values if x[0] == 0]


def decode_wrapper(t:int, frame_size:int, data: np.ndarray, wmrk: PatchworkWatermark):
    return wmrk.decode_single(data[t:t+frame_size])
