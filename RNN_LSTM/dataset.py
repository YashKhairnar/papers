import os
import random
from pathlib import Path

import torch 
import torch.nn.functional as F
import torchaudio
import torchaudio.functional
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, root, labels = None, sample_rate = 16000, seconds = 1.0, n_mels = 40, n_fft = 512, win_length = 400, hop_length = 100 ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * seconds )

        # Determine the labels ( folder names ) automatically if not provided
        all_folders = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        # exclude the background noise folder from labels
        if labels is None:
            self.labels = [ folder for folder in all_folders if not folder.startswith("_")]
        else:
            self.labels = labels
        self.label_to_index = { label: i for i,label in enumerate(self.labels)}

        #collect wav paths
        self.items = []
        for label in self.labels:
            for wav_path in (self.root/label).glob("*.wav"):
                    self.items.append((wav_path, self.label_to_index[label]))

        #audio transforms
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")


    #load wav
    def _load_wav(self, path: Path) -> torch.Tensor:
            wav, sr = torchaudio.load(str(path))
            wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            return wav
        
     # padding or trimming
    def _fixed_length(self, wav: torch.Tensor) -> torch.Tensor:
            if wav.shape[1] > self.num_samples:
                wav = wav[:, :self.num_samples]
            elif wav.shape[1] < self.num_samples:
                wav = F.pad(wav, (0, self.num_samples - wav.shape[1]))
            return wav
            

    def __len__(self):
        return len(self.items)

    def __getitem__(self,idx):
        wav_path, label = self.items[idx]
        wav = self._load_wav(wav_path)
        wav = self._fixed_length(wav)

        #wav to log-mel-spectrogram
        S = self.mel(wav)
        S_db = self.to_db(S)
        # S_db shape: (1, n_mels, time) -> squeeze -> (n_mels, time) -> transpose -> (time, n_mels)
        feats = S_db.squeeze(0).transpose(0, 1)
        # Normalize the features
        # The Tanh Problem: Vanilla RNNs use the tanh activation function. tanh is only "sensitive" (has a good gradient) near zero. If you feed it large negative numbers like -40, the output is nearly flat at -1.0.
        # Vanishing Gradients: When the function is flat, the gradient is zero. The model "stops learning" because it can't figure out how to adjust the weights to change the output.
        # The Fix: Instance Normalization scales each audio sample so its features have a mean of 0 and a standard deviation of 1. This "centers" your audio data right in the sweet spot of the RNN's activation function.
        feats = (feats - feats.mean()) / (feats.std() + 1e-6)

        length = feats.shape[0]
        return feats, label, length
