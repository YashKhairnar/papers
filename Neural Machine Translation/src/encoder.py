import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Bidirectional GRU Encoder.
    Processes the source sequence and returns all hidden states (outputs) 
    and the final combined hidden state.
    """
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Bidirectional GRU doubles the output hid_dim
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers,
                          bidirectional=True, batch_first=True, dropout=dropout)
        # fc layer reduces bidirectional output back to hid_dim for decoder alignment
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_len):
        embedded = self.dropout(self.embedding(x))
        # Use pack_padded_sequence for efficiency with variable length inputs
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'),
                                                  batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Combine forward and backward hidden states from the bidirectional GRU
        h_all = []
        for i in range(self.n_layers):
            combined = torch.cat((hidden[2*i,:,:], hidden[2*i+1,:,:]), dim=1)
            h_all.append(torch.tanh(self.fc(combined)))

        return outputs.permute(1, 0, 2), torch.stack(h_all)