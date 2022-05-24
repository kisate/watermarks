import torch
import torchaudio
import torchaudio.transforms as T
from data import LibriDataset
from torch.utils.data import DataLoader
from models import Conv2D
import torch.nn as nn

if __name__ == "__main__":
  libri_data = torchaudio.datasets.LIBRISPEECH("test_data", "dev-clean", download=True)


  print()
  print()
