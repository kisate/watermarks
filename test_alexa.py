import warnings

import matplotlib.pyplot as plt
from scipy.io import wavfile

from alexa import AmazonWatermark

import numpy as np

import scipy.signal

warnings.filterwarnings('error')

def calc_levels(vals):
    vals = np.array(vals)
    new_vals = []
    window_size = 200
    for i in range(vals.shape[0]):
        part = vals[i:i+window_size]

        rho_mean = part.mean()
        gamma = 3 * part.std()

        rho = (part[:10] - rho_mean).mean()
        new_vals.append(rho - gamma)

    return new_vals

if __name__ == "__main__":
    wav_fname = "/home/dumtrii/Documents/watermarks/africa-toto-22k.wav"

    samplerate, data = wavfile.read(wav_fname)

    data = data[:10000]
    # data = np.random.uniform(-5000, 5000, 3000)

    print(samplerate, data.shape)

    length = data.shape[0] / samplerate
    print(f"length = {length}s")

    print(len(data))

    watermark_length = int(samplerate * 0.1)

    wmrk = AmazonWatermark(6, 10, 1, 10)

    # for u in wmrk.watermark:
    #     for v in wmrk.watermark:
    #         print(u @ v)

    encoded = wmrk.encode(data, watermark_length, 100)
    # encoded = wmrk.encode_single(data[:watermark_length])

    # encoded = data

    # print(np.linalg.norm(data - encoded))
    #
    # print(data[(data - encoded) / (abs(data) + 1) > 0.3])

    # plt.plot(data)
    # plt.plot((data - encoded))
    # plt.show()


    # print(wmrk.calc_rho(encoded[:watermark_length]))

    import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()

    vals1 = wmrk.decode(encoded, watermark_length, 5, 400, 100)
    vals2 = wmrk.decode(data, watermark_length, 5, 400, 100)

    plt.plot(vals1)
    plt.plot(vals2)
    plt.show()

    vals1 = calc_levels(list(vals1))
    vals2 = calc_levels(list(vals2))

    plt.plot(vals1)
    plt.plot(vals2)
    plt.show()

    encoded_dec = scipy.signal.decimate(encoded, 4)
    data_dec = scipy.signal.decimate(data, 4)

    print("downsampled")

    vals1 = wmrk.decode(encoded_dec, watermark_length // 4, 5, 400, 100)
    vals2 = wmrk.decode(data_dec, watermark_length // 4 , 5, 400, 100)

    plt.plot(vals1)
    plt.plot(vals2)
    plt.show()

    vals1 = calc_levels(list(vals1))
    vals2 = calc_levels(list(vals2))

    plt.plot(vals1)
    plt.plot(vals2)
    plt.show()

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()

