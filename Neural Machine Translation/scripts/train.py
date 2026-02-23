import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Import our custom classes from the other files
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import MarathiTranslationDataset
from src.encoder import Encoder
from src.attention import Attention
from src.decoder import Decoder
from src.seq2seq import Seq2Seq
from src.eval_utils import calculate_bleu, translate_sentence


# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
EMBEDDING_DIM = 512
HIDDEN_DIM = 1024
N_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64          
LEARNING_RATE = 0.0005
N_EPOCHS = 20
CLIP = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

print(f"Using device: {DEVICE}")

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    
    # Calculate lengths for pack_padded_sequence
    src_lengths = torch.LongTensor([len(x) for x in src_batch])
    
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=0)
    
    return src_padded, trg_padded, src_lengths


def train_one_epoch(model, loader, optimizer, criterion, clip, scaler=None):
    model.train()
    epoch_loss = 0

    for i, (src, trg, src_len) in enumerate(loader):
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        optimizer.zero_grad()
        
        # --- Automatic Mixed Precision (AMP) for faster training ---
        # Note: CUDA only. MPS support is limited/experimental.
        use_amp = DEVICE.type == 'cuda'
        with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
            # The Seq2Seq model handles teacher forcing internally
            output = model(src, trg, src_len)

            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        epoch_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"  Batch [{i+1}/{len(loader)}] | Loss: {loss.item():.4f}")

    return epoch_loss / len(loader)

# ─── EVALUATION FUNCTION ──────────────────────────────────────────────────────
def evaluate(model, dataset, vocab, device, num_samples=100):
    model.eval()
    total_bleu = 0
    
    val_data = dataset.data
    num_samples = min(num_samples, len(val_data))
    
    for i in range(num_samples):
        item = val_data.iloc[i]
        src_text = item['english']
        trg_text = item['marathi']
        
        # 1. Get indices and translate
        src_indices = vocab.encode(f"<s>{src_text}</s>")
        prediction_tokens, _ = translate_sentence(src_indices, vocab, model, device)
        
        # 2. Decode subword tokens into a human-readable string
        # We need to know which indices were predicted. predict_sentence returns itos[idx]
        # Our translate_sentence actually returns itos[idx] list already.
        # We need to join them based on SentencePiece logic (sp.decode handles this).
        
        # Extract indices from tokens list for decoding
        pred_indices = [vocab.stoi[token] for token in prediction_tokens]
        if vocab.stoi["</s>"] in pred_indices:
            pred_indices = pred_indices[:pred_indices.index(vocab.stoi["</s>"])]
            
        prediction_str = vocab.decode(pred_indices).strip()
        
        # 3. Calculate BLEU (Word-level)
        # We split both by whitespace to get traditional word-level BLEU
        prediction_split = prediction_str.split()
        reference_split = trg_text.strip().split()
        
        total_bleu += calculate_bleu(reference_split, prediction_split)
        
    return total_bleu / num_samples

# ─── MAIN EXECUTION BLOCK ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading Marathi Translation dataset and SentencePiece tokenizer...")
    train_dataset = MarathiTranslationDataset(split='train') 
    valid_dataset_obj = MarathiTranslationDataset(split='test')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        # num_workers=4 if DEVICE.type != 'cpu' else 0,
        # pin_memory=True if DEVICE.type == 'cuda' else False
    )

    # ─── BUILD THE MODEL ──────────────────────────────────────────────────────────
    VOCAB_SIZE = len(train_dataset.vocab)
    print(f"Vocabulary Size: {VOCAB_SIZE}")

    attn = Attention(HIDDEN_DIM, HIDDEN_DIM)
    enc  = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec  = Decoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM, attn, DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    # --- Torch Compile (PyTorch 2.0+) ---
    if hasattr(torch, 'compile') and DEVICE.type != 'cpu':
        try:
            print("Compiling model for performance...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Skipping compilation: {e}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Gradient Scaler for AMP
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # --- Learning Rate Scheduler ---
    # Automatically reduces learning rate when progress stalls
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # --- RESUME FROM CHECKPOINT ---
    start_epoch = 0
    checkpoint_path = "checkpoint_epoch_latest.pt" # You can rename this to a specific epoch
    
    import os
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")

    print("\nStarting the training process...")
    for epoch in range(start_epoch, N_EPOCHS):
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CLIP, scaler=scaler)
        
        val_bleu = evaluate(model, valid_dataset_obj, train_dataset.vocab, DEVICE, num_samples=50)
        
        print(f"Average Loss: {train_loss:.4f}")
        print(f"Val BLEU: {val_bleu:.4f}")

        # Update learning rate based on training loss
        scheduler.step(train_loss)

        # --- CHECKPOINTING ---
        # Save after every epoch so you don't lose progress if you stop early
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'vocab': train_dataset.vocab
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, "checkpoint_epoch_latest.pt")
        print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pt & latest\n")

    torch.save(model.state_dict(), "bahdanau_marathi_final.pt")
    print("Training complete! Final model saved as bahdanau_marathi_final.pt")
