import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import Encoder
from src.attention import Attention
from src.decoder import Decoder
from src.seq2seq import Seq2Seq
from src.dataset import MarathiTranslationDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Configuration from inference.py/A100 Training
EMB_DIM = 256
HID_DIM = 1024
N_LAYERS = 4
DROPOUT = 0.3

# Dataset paths
CSV_PATH = "../data/samantar_dataset.csv"
SPM_PATH = "../models/en_mr_unigram.model"


# Use a placeholder for vocab size initially, or load it from the dataset
ds = MarathiTranslationDataset(CSV_PATH, SPM_PATH, split='test')
VOCAB_SIZE = len(ds.vocab)
print(f"Vocab Size: {VOCAB_SIZE}")

attn = Attention(HID_DIM, HID_DIM)
enc = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM, HID_DIM, attn, N_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, 'cpu')

params = count_parameters(model)
print(f"Total Trainable Parameters: {params:,}")
