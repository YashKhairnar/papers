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
| **Total Params**   | **221.8 Million**                                               |
| **Decoding**       | Greedy search, max 60 tokens                                    |

## Training Configuration

| Parameter           | Value                                      |
| :------------------ | :----------------------------------------- |
| **GPU**             | NVIDIA A100 80GB                           |
| **Dataset**         | Samantar En-Mr (filtered, 175,472 samples)    |
| **Max Seq Length**   | 128 tokens                                 |
| **Batch Size**      | 64                                         |
| **Optimizer**       | Adam (lr = 5e-4)                           |
| **LR Scheduler**    | ReduceLROnPlateau (factor=0.5, patience=2) |
| **Loss**            | CrossEntropyLoss (ignore `<pad>` index 0)  |
| **AMP**             | FP16 via `torch.amp` on CUDA               |
| **torch.compile**   | Enabled for optimized kernel execution     |
| **Gradient Clip**   | Max norm 1.0                               |
| **Epochs**          | 100                                        |
| **Train Time**      | ~5.5 Hours (A100 80GB)                     |
| **Evaluation**      | BLEU score (NLTK) + chrF (NLTK)            |
| **Teacher Forcing** | 100% during training                       |

## Performance Metrics

Evaluation conducted on 1,000 samples from the Samantar English-Marathi test set using the 100-epoch checkpoint.

| Metric | Score | Significance |
| :--- | :---: | :--- |
| **BLEU** | **4.54** | Measures exact word-level sequence match. |
| **chrF** | **28.66** | Character n-gram F-score; provides a robust measure for Marathi morphology. |

**Technical Analysis**: The pronounced margin between the chrF and BLEU scores suggests that while the model has acquired significant subword and character-level semantic knowledge, word-level exact matches remain challenging. This is typical for agglutinative languages like Marathi, where character-based metrics are often more representative of translation quality.

## Limitations & Known Constraints

While the model demonstrates strong foundational translation capabilities, users should be aware of the following architectural and dataset-driven limitations:

1. **Vocabulary Constraints (Out-of-Vocabulary)**: With a subword vocabulary size limited to 32,000 tokens, highly technical jargon, rare proper nouns, or domain-specific idioms may be tokenized sub-optimally, leading to literal or degraded translations.
2. **Agglutinative Morphology**: Marathi is highly agglutinative. While the chrF score indicates good character-level alignment, the model occasionally struggles with precise suffix attachments for complex tense/case combinations, resulting in grammatically near-misses (reflected in the lower BLEU score).
3. **Context Window & Optimal Input Length**: The model is trained on a maximum sequence length of 128 tokens, but its decoding phase utilizes a hard truncation at **60 tokens**. For optimal translation fidelity, inputs should be restricted to **1-2 sentences (under 40-50 words)**. Translations of lengthy paragraphs exceeding this window will be abruptly truncated and suffer from severe attention degradation.
4. **Domain Shift**: Trained exclusively on the Samantar corpus (which leans heavily toward news, government, and general web crawls), the model's performance may drop significantly on colloquial speech, slang, or raw social media text.

<p align="center">
  <img src="english_wordcloud.png" width="45%" alt="English Word Cloud">
  <img src="marathi_wordcloud.png" width="45%" alt="Marathi Word Cloud">
</p>

---

## Repository Structure

```
Neural Machine Translation/
├── app.py                             # Unified application (local dev + HF deployment)
├── train.py                           # Training and checkpointing logic
├── encoder.py                         # Bidirectional GRU encoder
├── decoder.py                         # GRU decoder with additive attention
├── attention.py                       # Bahdanau attention mechanism
├── seq2seq.py                         # Sequence-to-Sequence orchestration
├── dataset.py                         # Data processing and tokenization
├── eval_utils.py                      # Evaluation and metric computation
├── inference.py                       # Command-line inference and evaluation
├── best_a100_model_100.pt             # Trained model checkpoint (100 epochs)
├── samantar_dataset.csv               # Processed parallel corpus
└── requirements.txt                   # Dependency specification
```

## Model Weights

The pre-trained model checkpoint (`best_a100_model_100.pt`, 2.5GB) is available for download:
> 🔗 **[Download Weights (Google Drive)](https://drive.google.com/drive/folders/1cCHzMDJrPjBvI3YYztyKZ06l7SuErN7Y?usp=share_link)**

Place the downloaded `.pt` file in the root directory alongside `app.py`.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Application
```bash
python app.py
```

### 3. Execution of Evaluation Suite
To execute the automated evaluation on the test set:
```bash
python inference.py --eval --model best_a100_model_100.pt
```

## Architectural and Implementation Details

### Model Architecture
The system utilizes a 221.8M parameter Encoder-Decoder architecture featuring a Bahdanau-style Additive Attention mechanism.

#### Encoder
- **Architecture**: 4-Layer Bidirectional GRU with 1024 hidden units.
- **Contextualization**: Bidirectionality allows for a 2048-dimensional context vector, projected to 1024 dimensions via a dedicated linear layer to maintain dimensional consistency across the stack.
- **Optimization**: Sequence packing is implemented to mask padding tokens, ensuring the backward recurrent pass processes only valid tokens.

#### Decoder
- **Architecture**: 4-Layer Unidirectional GRU.
- **Attention Synergy**: The decoder utilizes a context vector derived from a weighted sum of encoder hidden states, calculated using the additive energy function: $v^T \tanh(W_a [s_{t-1} ; h_i])$.
- **Skip Connections**: Residual-like connections concatenate the recurrent state, context vector, and embedding at the output layer to mitigate gradient vanishing issues inherent in deep recurrent structures.

### Performance Optimization
- **Mixed Precision**: Leveraging `torch.amp.autocast` for 16-bit floating point precision provided a ~2.5x throughput increase on the A100 architecture.
- **Kernel Fusion**: `torch.compile` was utilized to fuse recurrent kernels, significantly reducing the overhead of CPU-GPU synchronization.
- **Data Filtering**: Outlier removal for sequences exceeding 600 characters bounded the memory complexity of the attention mechanism to $O(N \times M)$.

---
