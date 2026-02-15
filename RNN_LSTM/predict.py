
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from pathlib import Path
import argparse
import random

# Import local models
from VanillaRNN import VanillaRNN, LSTMModel

class Predictor:
    def __init__(self, model_path, model_type='LSTM', dataset_path=None, labels=None, device=None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 1. Load Labels
        if labels:
            self.labels = labels
        elif dataset_path:
            self.dataset_path = Path(dataset_path)
            all_folders = sorted([p.name for p in self.dataset_path.iterdir() if p.is_dir()])
            self.labels = [folder for folder in all_folders if not folder.startswith("_")]
        else:
            raise ValueError("Either labels or dataset_path must be provided")
            
        self.num_classes = len(self.labels)
        
        # 2. Initialize and Load Model
        if model_type == 'VanillaRNN':
            self.model = VanillaRNN(num_classes=self.num_classes)
        else:
            self.model = LSTMModel(num_classes=self.num_classes)
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 3. Preprocessing transforms (Must match dataset.py)
        self.sample_rate = 16000
        self.num_samples = 16000 
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=100,
            n_mels=40,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power")

    def preprocess(self, wav_path):
        # Load
        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.mean(dim=0, keepdim=True)
        
        # Resample
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            wav = resampler(wav)
            
        # Fix Length (Pad/Trim)
        if wav.shape[1] > self.num_samples:
            wav = wav[:, :self.num_samples]
        elif wav.shape[1] < self.num_samples:
            wav = F.pad(wav, (0, self.num_samples - wav.shape[1]))
            
        # Mel Spectrogram
        S = self.mel_transform(wav)
        S_db = self.to_db(S)
        
        # Format for RNN: (Time, Feats)
        feats = S_db.squeeze(0).transpose(0, 1)
        
        # Instance Norm
        feats = (feats - feats.mean()) / (feats.std() + 1e-6)
        
        return feats.unsqueeze(0).to(self.device), torch.tensor([feats.shape[0]]).to(self.device)

    def predict(self, wav_path, top_k=5):
        feats, lengths = self.preprocess(wav_path)
        with torch.no_grad():
            logits = self.model(feats, lengths)
            probs = F.softmax(logits, dim=1).squeeze(0)
            
            # Get top k predictions
            confidences, indices = torch.topk(probs, min(top_k, self.num_classes))
            
        # Return a dictionary suitable for Gradio's Label component
        result = {self.labels[idx.item()]: float(conf.item()) for conf, idx in zip(confidences, indices)}
        return result

def get_random_sample(dataset_path):
    dataset_path = Path(dataset_path)
    # Pick a random class folder
    labels = [p for p in dataset_path.iterdir() if p.is_dir() and not p.name.startswith("_")]
    chosen_label = random.choice(labels)
    # Pick a random wav file
    wav_files = list(chosen_label.glob("*.wav"))
    return random.choice(wav_files), chosen_label.name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['VanillaRNN', 'LSTM'])
    parser.add_argument('--file', type=str, help='Path to a wav file to predict')
    parser.add_argument('--random', action='store_true', help='Pick a random file from dataset and predict')
    args = parser.parse_args()

    predictor = Predictor(args.model_path, args.model_type, dataset_path='./dataset')

    if args.random:
        wav_path, true_label = get_random_sample('./dataset')
        print(f"Testing on random file: {wav_path}")
        print(f"Actual Label: {true_label}")
        pred, conf = predictor.predict(wav_path)
        print(f"Predicted: {pred} ({conf*100:.2f}% confidence)")
    elif args.file:
        pred, conf = predictor.predict(args.file)
        print(f"Predicted: {pred} ({conf*100:.2f}% confidence)")
    else:
        print("Please provide a --file path or use --random")
