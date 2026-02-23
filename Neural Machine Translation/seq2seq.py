import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    """
    Full Seq2Seq model containing the Encoder and Decoder with Attention.
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

    def forward(self, src, trg, src_len):
        """
        Forward pass that iterates through the target sequence.
        """
        batch_size, trg_len = src.shape[0], trg.shape[1]
        outputs = torch.zeros(trg_len, batch_size, self.decoder.fc_out.out_features).to(self.device)
        enc_out, hidden = self.encoder(src, src_len)
        mask = (src == 0).to(self.device)
        input = trg[:, 0]
        
        # Autoregressive decoding
        for t in range(1, trg_len):
            prediction, hidden, _ = self.decoder(input, hidden, enc_out, mask=mask)
            outputs[t] = prediction
            # Teacher Forcing: Feed the actual target token as the next input
            input = trg[:, t]
        return outputs

