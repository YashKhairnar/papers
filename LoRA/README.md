# LoRA Fine-Tuning: GPT-2 on IMDB Sentiment

A hands-on implementation of **Low-Rank Adaptation (LoRA)** applied to GPT-2 Small, fine-tuned for sentiment-conditioned text generation on the IMDB movie review dataset.

> Based on the paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)

---

## What is LoRA?

LoRA is a parameter-efficient fine-tuning (PEFT) technique. Instead of updating all weights of a pretrained model, it **freezes the original weights** and injects small trainable low-rank matrices (A and B) into specific layers:

```
W' = W + ΔW = W + B·A     where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << d,k
```

This drastically reduces the number of trainable parameters while preserving model quality.

---

## Project Structure

```
LoRA/
├── LoRA.ipynb                  # Main notebook (training, evaluation, comparison)
├── gpt2_finetuning_report.docx # Auto-generated results report
├── 2106.09685v2.pdf            # Original LoRA paper
└── README.md
```

---

## Setup

### Dependencies

```bash
pip install transformers datasets peft accelerate bitsandbytes trl
```

### Hardware

- Trained on: **NVIDIA T4 GPU** (Google Colab)
- Quantization: **4-bit NF4** via `bitsandbytes` (QLoRA-style)

---

## Experiment Configuration

| Hyperparameter     | Value                    |
|--------------------|--------------------------|
| Base Model         | GPT-2 Small (124M params)|
| Dataset            | IMDB (2,000 train examples) |
| LoRA Rank (`r`)    | 16                       |
| LoRA Alpha (`α`)   | 16                       |
| LoRA Dropout       | 0.05                     |
| Target Modules     | `c_attn`, `c_proj`       |
| Quantization       | 4-bit NF4 (double quant) |
| Epochs             | 10                       |
| Batch Size         | 40                       |
| Learning Rate      | 2e-4                     |
| LR Scheduler       | Linear warmup (20 steps) |
| Optimizer          | AdamW (weight decay 0.01)|
| Max Sequence Length| 256 tokens               |

### Trainable Parameters

```
Trainable params:   1,622,016
All params:       126,061,824
Trainable %:            1.29%
```

Only **1.29%** of the model's weights are updated during fine-tuning.

---

## Data Format

Each IMDB example is formatted with custom sentinel tokens to condition generation on sentiment:

```
<|sentiment|>POSITIVE<|review|>This movie is great...
```

The model learns to associate the sentiment token with the style of review text that follows.

---

## Training

### Loss Curve

| Epoch | Avg Loss |
|-------|----------|
| 1     | 3.9489   |
| 2     | 3.6025   |
| 3     | 3.5512   |
| 4     | 3.5315   |
| 5     | 3.5182   |
| 6     | 3.5151   |
| 7     | 3.5075   |
| 8     | 3.5035   |
| 9     | 3.5005   |
| 10    | 3.4981   |

The loss drops sharply in the first 3 epochs and plateaus — typical of fine-tuning a pretrained model on a small dataset.

---

## Results

### 1. Perplexity

Perplexity measures how "surprised" a model is by real review text. **Lower is better** — it means the model finds that style of writing more natural.

| Review Snippet | Base PPL | Fine-tuned PPL | Δ |
|---|---|---|---|
| *"This film is an absolute masterpiece..."* (positive) | 35.09 | 44.99 | +28.2% |
| *"Terrible movie. The plot made no sense..."* (negative) | 42.09 | 49.63 | +17.9% |
| *"A decent watch but nothing special..."* (neutral) | 116.28 | 126.30 | +8.6% |
| **Average** | — | — | **+18.2%** |

> **Note:** The fine-tuned model shows *higher* perplexity on raw un-prompted reviews — this is expected. It was trained on the structured `<|sentiment|>...<|review|>...` format, so plain review text feels less natural to it. The base model is a generalist over all web text.

---

### 2. Sentiment Accuracy (Keyword-based)

Each model was prompted 20 times (10 × POSITIVE, 10 × NEGATIVE) and outputs were scored via keyword matching.

| Model | Correct / Total | Accuracy |
|---|---|---|
| Base GPT-2 | 1 / 20 | 5% |
| **LoRA Fine-tuned** | **8 / 20** | **40%** |

> **8× improvement** in sentiment accuracy. The fine-tuned model learned to associate the `<|sentiment|>` token with the expected tone of the generated text.

---

### 3. Output Diversity (5 Generations, Same Prompt)

Prompt: `<|sentiment|>POSITIVE<|review|>`

| Model | Unique Vocab (5 runs) | Avg Words / Generation |
|---|---|---|
| Base GPT-2 | 85 | 17 |
| LoRA Fine-tuned | 169 | 47 |

The fine-tuned model generates **2× more unique vocabulary** and **2.8× longer** coherent outputs — a strong sign it has learned the IMDB review register.

---

### 4. Qualitative Comparison

**Prompt:** `<|sentiment|>POSITIVE<|review|>`

**Base GPT-2 output:**
```
RIGHT</a></span><br />
A few months ago I was in the midst of a conference call with an audience
member who asked me to take part in his discussion about "social justice."
The topic came up and, as usual—that is social justice. As my colleague had
pointed out before, most people would never hear any kind or allusions (but
that's ok.) Social issues are often considered taboo topics on this blog...
```
> The base model has no idea what `<|sentiment|>` means — it ignores it and generates unrelated text.

**LoRA Fine-tuned output:**
```
I've seen this film a few times before, and the first time I watched it was
as though everyone in my family were watching or listening to something. It
is not that people are offended by what they hear about movies like "Pulp
Fiction," but there's simply nothing offensive going on here with any sort
of material...
```
> On-topic, review-style prose. The model learned the format.

---

**Prompt:** `<|sentiment|>NEGATIVE<|review|>`

**Base GPT-2 output:**
```
COMMENT</title><meta http-equiv="Content-Type" content="text/html">
<a href="/wiki.php?f=/Mormonism_and_the_Religious_Discipline">
An Overview of Mormon Religion and the Religious Discipline (Part 1)...
```

**LoRA Fine-tuned output:**
```
I'm not a fan of this movie. The acting was atrocious, the characters were
ridiculous and it seemed like they should have been playing real people in
movies instead. "It's an insult to our society that such things are so
common." I don't understand why anyone would want more than $3 million for
something without even having heard about them on TV...
```

---

**Prompt:** `This movie was` (no special tokens — plain text)

**Base GPT-2:**
```
a great success. I loved it."
 - Mike McCallum, "The Last of Us"
(Photo: Warner Bros.)
From the original screenplay to his upcoming book The Man Who Killed Bill
Murray and How They Were Born in 1855; there are still many questions about
how this film will go down when released on BluRay & DVD...
```

**LoRA Fine-tuned:**
```
directed by the late Steve Buscemi, who has a long career as an actor and
director. The script is very well done with no sound effects which I found
difficult to hear during rehearsals because he had so much music going on
there. The film takes place just after WWII but shows some character
development throughout history — including several characters from previous
films such as Asparagus & Bulldog (1953) through Jaws: A Life After Death...
```
> Both models handle plain prompts reasonably well. The fine-tuned model naturally leans toward review-style language even without the sentinel tokens.

---

## Key Takeaways

- **1.29% of parameters** trained — yet the model's output style changed dramatically.
- Sentiment accuracy jumped from **5% → 40%** (8× improvement) with LoRA fine-tuning.
- LoRA successfully taught GPT-2 to **follow custom prompt structure** (`<|sentiment|>` / `<|review|>` tokens) — the base model ignores them entirely.
- The fine-tuned model produces **longer, more coherent, domain-specific text**.
- Higher perplexity on raw text is expected — the model specialised to a structured format, trading generality for domain fit.
- 2,000 examples is enough to see clear behavioral change on GPT-2 Small with LoRA.

---

## References

- Hu, E. J., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *arXiv:2106.09685*
- Dettmers, T., et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- HuggingFace PEFT library: https://github.com/huggingface/peft
