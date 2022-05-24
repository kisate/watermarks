import warnings

import matplotlib.pyplot as plt
import scipy.signal
from scipy.io import wavfile

from patchwork import PatchworkWatermark

import numpy as np


warnings.filterwarnings('error')

if __name__ == "__main__":
    wav_fname = "/home/dumtrii/Documents/watermarks/africa-toto-22k.wav"

    samplerate, data = wavfile.read(wav_fname)

    data = data[samplerate:samplerate*3]
    wavfile.write("orig.wav", samplerate, data)
    # data = np.random.uniform(-5000, 5000, 3000)

    print(samplerate, data.shape)

    length = data.shape[0] / samplerate
    print(f"length = {length}s")

    print(len(data))

    watermark_length = 1000

    wmrk = PatchworkWatermark(45, alpha=40000)

    # for u in wmrk.watermark:
    #     for v in wmrk.watermark:
    #         print(u @ v)

    encoded = wmrk.encode(data, 10)
    wavfile.write("encoded.wav", samplerate, encoded)
    # encoded = wmrk.encode_single(data[:watermark_length])

    # encoded = data

    # print(np.linalg.norm(data - encoded))
    #
    # print(data[(data - encoded) / (abs(data) + 1) > 0.3])

    plt.plot(data)
    plt.plot((data - encoded))
    plt.show()


    # print(wmrk.calc_rho(encoded[:watermark_length]))

    # import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()

    vals1 = wmrk.decode(encoded, 10)

    print(vals1)

    plt.plot(vals1)
    plt.show()

    # encoded = scipy.signal.decimate(encoded, 2)
    #
    # print("downsampled")
    #
    # vals2 = wmrk.decode(encoded)
    #
    # plt.show()

    # vals2 = wmrk.decode(data, watermark_length, 5, 400, 100)

    # plt.plot(vals1)
    # plt.plot(vals2)
    # plt.show()

    # vals1 = calc_levels(list(vals1))
    # vals2 = calc_levels(list(vals2))
    #
    # plt.plot(vals1)
    # plt.plot(vals2)
    # plt.show()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
