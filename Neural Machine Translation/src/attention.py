import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Bahdanau Attention (Additive Attention) logic:
    1. Take the current Decoder hidden state (s) and ALL Encoder hidden states (h).
    2. Feed them into a small Neural Network to get an "Alignment Score".
    3. Softmax the scores to get weights that sum to 1.0.
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        
        # self.attn is the "W" in the paper. It combines enc_hidden ( h ) and dec_hidden ( s ).
        # encoder_hidden_dim is hidden_dim * 2 (concatenated Bi-GRU states)
        self.attn = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        
        # self.v is the final vector that collapses the energy into a single score.
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden shape: [batch_size, dec_hidden_dim]
        # encoder_outputs shape: [src_len, batch_size, enc_hidden_dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # 1. Repeat the decoder's hidden state for every source word
        # hidden is [1, batch, dec_hid] or [batch, dec_hid]
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        
        # [batch, dec_hid] -> [src_len, batch, dec_hid]
        hidden = hidden.repeat(src_len, 1, 1)
        
        # 2. Calculate "Energy" (how well each source word matches current state)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        # energy shape: [src_len, batch, dec_hid]
        
        attention = self.v(energy).squeeze(2).permute(1, 0)
        # attention shape: [batch, src_len]
        
        # --- MASKING ---
        # If a mask is provided, set the energy of padding tokens to a very small number.
        # This makes the softmax output for those tokens essentially 0.
        if mask is not None:
            # mask: [batch, src_len] where 1 is mask, 0 is real
            attention = attention.masked_fill(mask == 1, -1e4)

        # 4. Use Softmax to turn scores into probabilities (sum to 1)
        return torch.softmax(attention, dim=1)
