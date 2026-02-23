# English → Marathi Neural Machine Translation

A sequence-to-sequence model with **Bahdanau (Additive) Attention** for English-to-Marathi translation, trained on the [Samantar](https://ai4bharat.iitm.ac.in/samantar/) parallel corpus using an NVIDIA A100 80GB GPU.

> **Paper**: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Bahdanau et al., 2014

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Seq2Seq Model                          │
│                                                             │
│  ┌──────────────────┐       ┌───────────────────────┐       │
│  │     Encoder      │──────▶│       Decoder         │       │
│  │  4-Layer BiGRU   │       │   4-Layer GRU         │       │
│  │  (Bidirectional) │       │   + Additive Attention│       │
│  └──────────────────┘       └───────────────────────┘       │
│           ▲                           │                     │   
│     SentencePiece              SentencePiece                │
│       Encode                     Decode                     │
└─────────────────────────────────────────────────────────────┘
```

| Component          | Detail                                                          |
| :----------------- | :-------------------------------------------------------------- |
| **Encoder**        | 4-layer Bidirectional GRU, 1024 hidden units, packed sequences  |
| **Decoder**        | 4-layer Unidirectional GRU, 1024 hidden units, teacher forcing  |
| **Attention**      | Additive (Bahdanau) — `v · tanh(W[h_enc; h_dec])`              |
| **Embedding**      | 256-dim learned embeddings (inference) / 512-dim (training)     |
| **Tokenizer**      | SentencePiece Unigram (shared En-Mr vocabulary)                 |
| **Bridge**         | Linear `enc_hid*2 → dec_hid` with tanh (per layer)             |
| **Dropout**        | 0.3 across all layers                                           |
| **Decoding**       | Greedy search, max 60 tokens                                    |

## Training Configuration

| Parameter           | Value                                      |
| :------------------ | :----------------------------------------- |
| **GPU**             | NVIDIA A100 80GB                           |
| **Dataset**         | Samantar En-Mr (filtered, 100K samples)    |
| **Max Seq Length**   | 128 tokens                                 |
| **Batch Size**      | 64                                         |
| **Optimizer**       | Adam (lr = 5e-4)                           |
| **LR Scheduler**    | ReduceLROnPlateau (factor=0.5, patience=2) |
| **Loss**            | CrossEntropyLoss (ignore `<pad>` index 0)  |
| **AMP**             | FP16 via `torch.amp` on CUDA               |
| **torch.compile**   | Enabled for optimized kernel execution     |
| **Gradient Clip**   | Max norm 1.0                               |
| **Epochs**          | 60                                         |
| **Evaluation**      | BLEU score (NLTK, smoothing method 1)      |
| **Teacher Forcing** | 100% during training                       |

## Repository Structure

```
Neural Machine Translation/
├── gr_app.py                          # Gradio web UI with attention heatmaps (Plotly)
├── train.py                           # Training loop (AMP, checkpointing, BLEU eval)
├── encoder.py                         # Bidirectional GRU encoder with packed sequences
├── decoder.py                         # GRU decoder with context + attention
├── attention.py                       # Bahdanau additive attention mechanism
├── seq2seq.py                         # Seq2Seq orchestrator (teacher forcing)
├── dataset.py                         # PyTorch Dataset + SentencePieceWrapper
├── eval_utils.py                      # BLEU scoring + greedy translate utility
├── inference.py                       # Inference & test evaluation CLI
├── best_a100_model_80.pt              # Best checkpoint (80 epochs)
├── samantar_dataset.csv               # Parallel corpus (English-Marathi)
└── requirements.txt                   # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Web UI

```bash
python gr_app.py
```

Opens at `http://127.0.0.1:7860` — type English text or click a sample to translate. An interactive **attention heatmap** is generated for each translation.

### 3. Evaluate on Test Set
Run the model on 1,000 samples from the test split and calculate objective metrics:
```bash
python inference.py --eval --model best_a100_model_80.pt
```

## Key Implementation Details

### Attention Mechanism (`attention.py`)
Additive (Bahdanau) attention computes alignment scores:
```
energy = v · tanh(W_attn · [h_encoder; h_decoder])
attention = softmax(energy)
```
Padding tokens are masked with `-1e4` before softmax to prevent the model from attending to `<pad>`.

### Packed Sequences (`encoder.py`)
The encoder uses `pack_padded_sequence` / `pad_packed_sequence` to skip `<pad>` tokens during RNN computation. This is critical for the backward GRU to produce correct representations.

### Teacher Forcing (`seq2seq.py`)
During training, the decoder receives the **ground-truth** previous token at each step (100% teacher forcing). At inference time (`gr_app.py`), greedy decoding is used — the model feeds its own predictions back as input.

### Checkpoint Resumption (`train.py`)
Saves `model_state_dict`, `optimizer_state_dict`, epoch, loss, and vocab every epoch. On restart, automatically loads the latest checkpoint and resumes training.

### Web UI (`gr_app.py`)
Self-contained Gradio app that bundles the model architecture, loads the checkpoint, and provides:
- Real-time En→Mr translation with greedy decoding
- Interactive Plotly attention heatmap showing source-target alignment
- 5 pre-loaded test samples (simple → complex)

---

*Research exploration into NMT optimization for Indic languages • 2026*
