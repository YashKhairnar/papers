import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from dataset import AudioDataset


class VanillaRNN(nn.Module):
    def __init__(self, input_dim = 40, hidden_dim = 128, num_layers = 2, num_classes = 35, bidirectional = False, dropout = 0.2):
        super().__init__()
        self.rnn = nn.RNN(
            input_size = input_dim,
            hidden_size = hidden_dim,
            nonlinearity = "tanh", # activation function
            batch_first = True,
            num_layers = num_layers,
            bidirectional = bidirectional,
            dropout = dropout if num_layers > 1 else 0.0
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.ln = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x, lengths):
        # x: (batch, time, features)
        # lengths need to be on CPU for pack_padded_sequence usually
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed) # We ignore the output at each timestep (_) and only keep the final hidden state (h)
        hidden = h[-1]
        #Layer Normalization re-normalizes the hidden states at every time step. 
        # It ensures that no matter how many time steps you have, the activations inside the RNN stay within a healthy range. 
        # It makes the "loss landscape" much smoother  
        hidden = self.ln(hidden)
        return self.fc(hidden)


class LSTMModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim = 128, num_layers = 2, num_classes = 35, bidirectional = False, dropout = 0.2 ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            dropout = dropout,
            batch_first = True,
            bidirectional = False,
        )
        out_dim = hidden_dim
        self.ln = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x, length):
        packed = pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed)
        hidden = h[-1]
        hidden = self.ln(hidden)
        return self.fc(hidden)


def train_step(model, batch, optimizer, device='mps'):
    # batch unpacks to: feats, label, length
    x, y, lengths = batch
    x, y, lengths = x.to(device), y.to(device), lengths.to(device)
    
    optimizer.zero_grad()
    # forward pass
    logits = model(x, lengths)
    # compute loss
    loss = F.cross_entropy(logits, y)
    # backward pass
    loss.backward()

    # vanilla RNNs can explode gradients → clipping helps a lot
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # update weights
    optimizer.step()
    acc = (logits.argmax(dim=1) == y ).float().mean().item()

    return loss.item(), acc


def validate(model, val_loader, device='mps'):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y, lengths = batch
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            logits = model(x, lengths)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            total_acc += (logits.argmax(dim=1) == y).float().mean().item()
    return total_loss / len(val_loader), total_acc / len(val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='VanillaRNN', choices=['VanillaRNN', 'LSTM'])
    args = parser.parse_args()

    writer = SummaryWriter("runs/lstm")
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    dataset_path = "/Users/yash/Desktop/papers/RNN_LSTM/dataset"

    try:
        # Load full dataset
        full_dataset = AudioDataset(dataset_path)
        
        # Split into train (80%) and validation (20%)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
        
        print(f"Dataset split: {train_size} training, {val_size} validation samples")
        
        num_classes = len(full_dataset.labels)
        print(f"Number of classes: {num_classes}")
        
        if args.model == 'VanillaRNN':
                model = VanillaRNN(num_classes=num_classes).to(device)
        elif args.model == 'LSTM':
            model = LSTMModel(num_classes=num_classes).to(device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_loss = float('inf')

        for epoch in range(100):
            # Training phase
            model.train()
            epoch_loss = 0
            epoch_acc = 0
            for batch_idx, batch in enumerate(train_loader):
                loss, acc = train_step(model, batch, optimizer, device=device)
                epoch_loss += loss
                epoch_acc += acc
            
            # Calculate average loss and accuracy for the epoch
            avg_train_loss = epoch_loss / len(train_loader)
            avg_train_acc = epoch_acc / len(train_loader)
            
            print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Accuracy/train", avg_train_acc, epoch)
            
            # Validation phase
            val_loss, val_acc = validate(model, val_loader, device=device)
            print(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_acc, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
        # Save final model
        torch.save(model.state_dict(), "final_model.pth")
        print("Training complete! Saved 'final_model.pth'")
        writer.close()
    except Exception as e:
        print(f"An error occurred: {e}")