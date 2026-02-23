import torch
import torch.nn as nn
import sentencepiece as spm
import plotly.graph_objects as go
import gradio as gr
import os

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear((enc_hid * 2) + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2).permute(1, 0)
        if mask is not None:
            attention = attention.masked_fill(mask == 1, -1e4)
        return torch.softmax(attention, dim=1)


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
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers,
                          bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_len):
        embedded = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'),
                                                   batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h_all = [torch.tanh(self.fc(torch.cat((hidden[2*i,:,:], hidden[2*i+1,:,:]), dim=1)))
                 for i in range(self.n_layers)]
        return outputs.permute(1, 0, 2), torch.stack(h_all)


class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, enc_hid, dec_hid, attention, n_layers, dropout):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid * 2) + emb_dim, dec_hid, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear((enc_hid * 2) + dec_hid + emb_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(input.unsqueeze(0)))
        a = self.attention(hidden[-1:], encoder_outputs, mask=mask).unsqueeze(1)
        context = torch.bmm(a, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
        output, hidden = self.rnn(torch.cat((embedded, context), dim=2), hidden)
        prediction = self.fc_out(torch.cat((output.squeeze(0), context.squeeze(0), embedded.squeeze(0)), dim=1))
        return prediction, hidden, a.squeeze(1)


class Seq2Seq(nn.Module):
    """
    Full Seq2Seq model containing the Encoder and Decoder with Attention.
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device


class SentencePieceWrapper:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id, self.bos_id, self.eos_id = [
            self.sp.piece_to_id(x) for x in ['<pad>', '<s>', '</s>']
        ]

    def encode(self, text):  return self.sp.encode(text)
    def decode(self, ids):   return self.sp.decode(ids)
    def __len__(self):       return self.sp.get_piece_size()


# ---------------------------------------------------------------------------
# Engine Load (once at startup)
# ---------------------------------------------------------------------------
# Auto-detect best available checkpoint:
#   - Local dev:  uses full-precision 100-epoch model
#   - HF Space:   uses FP16-compressed model (413 MB)
CKPT_PATH = next(
    (f for f in ["best_a100_model_100.pt", "deploy_model_fp16.pt"] if os.path.exists(f)),
    "deploy_model_fp16.pt"  # fallback (will show "not found" warning)
)
VOCAB_PATH = "en_mr_unigram.model"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    vocab = SentencePieceWrapper(VOCAB_PATH)
    V = len(vocab)
    attn = Attention(1024, 1024)
    enc  = Encoder(V, 256, 1024, 4, 0.3)
    dec  = Decoder(V, 256, 1024, 1024, attn, 4, 0.3)
    model = Seq2Seq(enc, dec, device).to(device)

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        ENGINE_STATUS = "✅ Model loaded successfully"
    else:
        ENGINE_STATUS = "⚠️ Checkpoint not found — using untrained weights"
except Exception as e:
    model = vocab = None
    ENGINE_STATUS = f"❌ Error loading model: {e}"

print(ENGINE_STATUS)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def translate_sentence(sentence, max_len=200):
    if model is None or vocab is None:
        return "Model not available.", None

    model.eval()
    src_ids = [vocab.bos_id] + vocab.encode(sentence) + [vocab.eos_id]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    src_len    = torch.LongTensor([len(src_ids)])

    with torch.no_grad():
        enc_out, hidden = model.encoder(src_tensor, src_len)

    trg_ids    = [vocab.bos_id]
    attentions = torch.zeros(max_len, 1, len(src_ids)).to(device)

    for t in range(max_len):
        trg_tensor = torch.LongTensor([trg_ids[-1]]).to(device)
        with torch.no_grad():
            pred, hidden, attention = model.decoder(
                trg_tensor, hidden, enc_out, mask=(src_tensor == 0).to(device)
            )
        attentions[t] = attention
        idx = pred.argmax(1).item()
        if idx == vocab.eos_id:
            break
        trg_ids.append(idx)

    res = [i for i in trg_ids if i not in [vocab.bos_id, vocab.eos_id, vocab.pad_id]]
    translation = vocab.decode(res)
    attention_data = attentions[:len(trg_ids) - 1]
    return translation, attention_data


def build_heatmap(input_text, translation, attention_data):
    att_numpy  = attention_data.squeeze(1).cpu().detach().numpy()
    input_ids  = vocab.encode(input_text)
    src_pieces = [vocab.sp.id_to_piece(idx) for idx in input_ids]
    src_tokens = ['<sos>'] + src_pieces + ['<eos>']
    trg_tokens = translation.split() + ['<eos>']

    fig = go.Figure(data=go.Heatmap(
        z=att_numpy,
        x=src_tokens,
        y=trg_tokens,
        colorscale=[[0.0, '#f8fafc'], [0.2, '#e0e7ff'],
                    [0.5, '#818cf8'], [0.8, '#4f46e5'], [1.0, '#312e81']],
        showscale=True,
        hovertemplate='<b>%{x}</b> → <b>%{y}</b><br>Focus: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="Neural Alignment Map", font=dict(size=14, color="#64748b",
                   family="Plus Jakarta Sans"), x=0.5),
        xaxis=dict(side="bottom", tickangle=45, showgrid=False,
                   tickfont=dict(color="#1e293b", size=11)),
        yaxis=dict(autorange="reversed", showgrid=False,
                   tickfont=dict(color="#1e293b", size=11)),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=50, b=40, l=40, r=40),
        height=520,
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig


def run_translation(input_text):
    if not input_text or not input_text.strip():
        return "", None, "Please enter some text first."

    translation, attention_data = translate_sentence(input_text.strip())

    if attention_data is None:
        return translation, None, ENGINE_STATUS

    fig = build_heatmap(input_text.strip(), translation, attention_data)
    return translation, fig, ENGINE_STATUS


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

.gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Main container layout */
#main-container { max-width: 1400px !important; margin: 0 auto !important; padding: 1.5rem !important; }

/* Status text */
.status-md p { font-size: 0.85rem !important; color: #64748b !important; margin: 0 !important; font-weight: 500; }

/* Subtitle */
.subtitle { color: #64748b !important; font-size: 1.1rem !important; margin-bottom: 2rem !important; }

/* Sample Sidebar Styling */
.sample-sidebar {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
}

/* Scrollable area for buttons */
.sample-list {
    overflow-y: auto !important;
    max-height: 540px !important;
    padding-right: 0.75rem !important;
    margin-top: 1rem !important;
}

/* Premium Sample Buttons */
.sample-btn {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #334155 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    text-align: left !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
    margin-bottom: 0.5rem !important;
}

.sample-btn:hover {
    border-color: #4f46e5 !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.12) !important;
    color: #4f46e5 !important;
    transform: translateY(-2px) !important;
    background-color: #f5f3ff !important;
}
"""

SAMPLES = [
    "Where's the nearest train station?",
    "I like to read books in free time.",
    "What are you thinking about?",
    "What's your name?",
    "There is a puddle on the road.",
]

# Use robust Gradio 6 native theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    spacing_size="sm",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"]
)

with gr.Blocks(title="Neural Lingua — En→Mr NMT") as demo:

    with gr.Column(elem_id="main-container"):
        gr.Markdown("""
# English → Marathi Translator
<p class="subtitle">Neural Machine Translation powered by BiGRU + Additive Attention</p>
""")
        
        # ── Main Layout (Flexbox for equal heights) ──────────────────
        with gr.Row(elem_id="equal-height-row"):
            
            # LEFT: Samples (1/3 width)
            with gr.Column(scale=1, min_width=250, elem_classes="sample-sidebar"):
                gr.Markdown("### ⚡ Quick Test Samples")
                sample_btns = []
                for text in SAMPLES:
                    btn = gr.Button(text, elem_classes="sample-btn")
                    sample_btns.append((btn, text))

            # RIGHT: Translator (2/3 width)
            with gr.Column(scale=2, elem_classes="translator-col"):
                with gr.Row():
                    input_box = gr.Textbox(
                        placeholder="Type English text here, or pick a sample →",
                        lines=8,
                        label="English Source (Input)"
                    )
                    output_box = gr.Textbox(
                        lines=8,
                        label="Marathi Translation (Output)",
                        interactive=False
                    )

                with gr.Row():
                    translate_btn = gr.Button("Translate", variant="primary", size="lg")
                
                with gr.Row():
                    status_bar = gr.Markdown(ENGINE_STATUS, elem_classes="status-md")

        # ── Heatmap ──────────────────────────────────────────────────
        gr.HTML("<hr class='divider'>")
        heatmap_plot = gr.Plot(label="Neural Alignment Analysis", visible=False)

        # ── Wire sample buttons → input ──────────────────────────────
        for btn, text in sample_btns:
            btn.click(fn=lambda t=text: t, outputs=input_box)

        # ── Translation logic ────────────────────────────────────────
        def on_translate(text):
            translation, fig, status = run_translation(text)
            show_heatmap = fig is not None
            return translation, gr.update(value=fig, visible=show_heatmap), status

        translate_btn.click(fn=on_translate, inputs=[input_box],
                            outputs=[output_box, heatmap_plot, status_bar])
        input_box.submit(fn=on_translate, inputs=[input_box],
                         outputs=[output_box, heatmap_plot, status_bar])

        gr.HTML("""
<div style="text-align:center; margin-top:2rem; color:#cbd5e1; font-size:0.72rem; padding-bottom:1rem;">
  Neural Lingua NMT Research &bull; 2026
</div>
""")

if __name__ == "__main__":
    demo.launch(share=False, show_error=True, css=CUSTOM_CSS, theme=theme)
