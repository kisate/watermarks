import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torchaudio
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

from tqdm import tqdm

class LibriDataset(Dataset):
    def __init__(self, data_path, secret_size=100, n_mels=400, n_frames=400):
        self.data_path = Path(data_path)
        self.secret_size = secret_size
        self.n_mels = n_mels
        self.n_frames = n_frames // 2

        n_fft = 256
        win_length = None
        hop_length = 128

        self.sample_size = n_fft * self.n_frames - n_fft // 2
        self.files_list = list(self.data_path.rglob("*.flac"))[:10]
        self.file_lengths = np.array(list(map(lambda x: torchaudio.load(x)[0].shape[1], tqdm(self.files_list))), dtype=np.int32)
        self.files_list = [x for i, x in enumerate(self.files_list) if self.file_lengths[i] > self.sample_size]
        self.file_lengths = np.array([x for x in self.file_lengths if x > self.sample_size])
        self.file_lengths = np.floor_divide(self.file_lengths, self.sample_size)
        self.cum_file_lengths = np.cumsum(self.file_lengths)



        self.spectrogram = T.MelSpectrogram(
            # sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        )

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cum_file_lengths, idx)
        file_path = self.files_list[file_idx]

        waveform, _ = torchaudio.load(file_path)
        waveform = waveform[0, :]
        part_idx = self.file_lengths[file_idx] - (self.cum_file_lengths[file_idx] - idx)

        waveform = waveform[part_idx * self.sample_size:(part_idx + 1) * self.sample_size]
        spectrogram = self.spectrogram(waveform)

        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()

        return spectrogram.unsqueeze(0), secret

    def __len__(self):
        return self.cum_file_lengths[-1]


if __name__ == '__main__':
    from models import transform_net

    dataset = LibriDataset(data_path='data/LibriSpeech/dev-clean', secret_size=100, n_frames=128, n_mels=64)
    dataloader = DataLoader(dataset, batch_size=4)
    image_input, secret_input = next(iter(dataloader))
    print(image_input.shape, secret_input.shape)
    print(image_input.max())

    import models

    encoder = models.StegaStampEncoder(n_frames=128, height=64)
    decoder = models.StegaStampDecoder(100)

    res = encoder((secret_input, image_input))
    print(res.shape)
    res = decoder(image_input + res)
    print(res.shape)

