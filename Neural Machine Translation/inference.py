import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import argparse
from dataset import MarathiTranslationDataset
from encoder import Encoder
from attention import Attention
from decoder import Decoder
from seq2seq import Seq2Seq
from eval_utils import translate_sentence, run_test_evaluation

def display_attention(sentence_pieces, translation_pieces, attention):
    """
    Plots the attention heatmap using Matplotlib.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    
    # attention shape: [trg_len, src_len]
    attention = attention.cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
    
    # Set up axes
    ax.tick_params(labelsize=12)
    
    # x-axis is the source (input) sentence
    ax.set_xticklabels([''] + sentence_pieces, rotation=45)
    # y-axis is the predicted translation
    # Note: Use a font that supports Devanagari if available
    ax.set_yticklabels([''] + translation_pieces)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.xlabel('Source (English)')
    plt.ylabel('Prediction (Marathi)')
    plt.savefig("attention_map.png")
    print("Attention map saved as attention_map.png")

def load_model(model_path, vocab_size, device):
    # Hyperparameters (must match train.py / notebook)
    EMB_DIM = 256
    HID_DIM = 1024
    N_LAYERS = 4
    DROPOUT = 0.3
    
    attn = Attention(HID_DIM, HID_DIM)
    enc = Encoder(vocab_size, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(vocab_size, EMB_DIM, HID_DIM, HID_DIM, attn, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Handle torch.compile prefix (_orig_mod.)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="NMT Inference and Evaluation Script")
    parser.add_argument("--sentence", type=str, help="Sentence to translate")
    parser.add_argument("--eval", action="store_true", help="Run full test set evaluation")
    parser.add_argument("--model", type=str, default="best_a100_model_80.pt", help="Path to model checkpoint")
    parser.add_argument("--csv", type=str, default="samantar_dataset.csv", help="Path to Samantar CSV")
    parser.add_argument("--spm", type=str, default="en_mr_unigram.model", help="Path to SentencePiece model")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found.")
        return

    print(f"Loading dataset and model...")
    # Use small subset for quick loading of vocab if just translating one sentence
    ds = MarathiTranslationDataset(args.csv, args.spm, split='test')
    model = load_model(args.model, len(ds.vocab), device)

    if args.eval:
        run_test_evaluation(model, ds, device, num_samples=1000)
    elif args.sentence:
        src_ids = [ds.vocab.bos_id] + ds.vocab.encode(args.sentence) + [ds.vocab.eos_id]
        translation_pieces, attention = translate_sentence(src_ids, ds.vocab, model, device)
        
        translation = ds.vocab.decode(ds.vocab.sp.piece_to_id(p) for p in translation_pieces)
        print(f"\nSource: {args.sentence}")
        print(f"Translation: {translation}")
        
        # Visualize
        src_pieces = [ds.vocab.itos[idx] for idx in src_ids]
        display_attention(src_pieces, translation_pieces, attention)
    else:
        print("Please provide a --sentence or use --eval flag.")

if __name__ == "__main__":
    main()

