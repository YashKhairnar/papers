import sentencepiece as spm
import pandas as pd
import torch
from torch.utils.data import Dataset

class SentencePieceWrapper:
    """
    A convenience wrapper around SentencePieceProcessor to handle tokenization
    and special token management (PAD, BOS, EOS).
    """
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Standard SentencePiece IDs: PAD=0, UNK=1, BOS=2, EOS=3
        self.pad_id = self.sp.piece_to_id('<pad>')
        self.bos_id = self.sp.piece_to_id('<s>')
        self.eos_id = self.sp.piece_to_id('</s>')
        
        # Convenience mappings for token/index conversion
        self.itos = {i: self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())}
        self.stoi = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}

    def encode(self, text): 
        """Encodes raw text into a list of subword integer IDs."""
        return self.sp.encode(text)

    def decode(self, ids): 
        """Decodes subword integer IDs back into a single string."""
        return self.sp.decode(ids)

    def __len__(self): 
        """Returns the total size of the vocabulary."""
        return self.sp.get_piece_size()

class MarathiTranslationDataset(Dataset):
    """
    PyTorch Dataset for English-Marathi translation.
    Handles data cleaning, length filtering, and tokenization with start/end tokens.
    """
    def __init__(self, csv_path, model_path, split='train', split_ratio=0.98, max_len=128):
        df = pd.read_csv(csv_path)
        self.vocab = SentencePieceWrapper(model_path)

        # Clean data: Remove rows with missing translations
        df = df.dropna(subset=['src', 'tgt'])
        df['src'] = df['src'].astype(str)
        df['tgt'] = df['tgt'].astype(str)

        if split == 'train':
            # Remove extreme outliers to stabilize training and save memory
            print("Filtering dataset for length...")
            df = df[(df['src'].str.len() < 600) & (df['tgt'].str.len() < 600)]

        # Split into training and validation sets
        train_size = int(len(df) * split_ratio)
        self.data = df.iloc[:train_size].reset_index(drop=True) if split == 'train' else df.iloc[train_size:].reset_index(drop=True)
        print(f"{split.capitalize()} set size: {len(self.data)}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        """
        Returns source and target tensors with BOS and EOS tokens added.
        """
        row = self.data.iloc[idx]
        src_ids = [self.vocab.bos_id] + self.vocab.encode(row['src']) + [self.vocab.eos_id]
        trg_ids = [self.vocab.bos_id] + self.vocab.encode(row['tgt']) + [self.vocab.eos_id]
        return torch.tensor(src_ids), torch.tensor(trg_ids)