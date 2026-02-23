import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Attention-based GRU Decoder.
    Predicts the next token given the current token, the hidden state, and context.
    """
    def __init__(self, out_dim, emb_dim, enc_hid, dec_hid, attention, n_layers, dropout):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(out_dim, emb_dim)
        # RNN input includes both the target embedding and the context vector from attention
        self.rnn = nn.GRU((enc_hid * 2) + emb_dim, dec_hid, num_layers=n_layers, dropout=dropout)
        # Output layer predicts over the vocabulary
        self.fc_out = nn.Linear((enc_hid * 2) + dec_hid + emb_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(input.unsqueeze(0)))
        # Calculate attention weights
        a = self.attention(hidden[-1:], encoder_outputs, mask=mask).unsqueeze(1)
        # Generate context vector: dot product of attention weights and encoder outputs
        context = torch.bmm(a, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Predict next token by combining GRU output, context, and current embedding
        prediction = self.fc_out(torch.cat((output.squeeze(0), context.squeeze(0), embedded.squeeze(0)), dim=1))
        return prediction, hidden, a.squeeze(1)

