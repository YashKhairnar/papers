
import gradio as gr
from predict import Predictor, get_random_sample
from pathlib import Path
import torch

# Configuration
MODEL_PATH = "best_model.pth"
MODEL_TYPE = "LSTM"
LABELS = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

# Initialize the predictor
predictor = Predictor(MODEL_PATH, MODEL_TYPE, labels=LABELS)

# Get the supported words early
vocab = ", ".join(predictor.labels)

def classify_audio(audio_path):
    if audio_path is None:
        return None
    
    # Returns dictionary of top predictions for gr.Label
    results = predictor.predict(audio_path, top_k=5)
    return results

# Create a truly "Premium" Theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    block_title_text_weight="600",
    container_radius="12px",
    button_large_radius="8px",
)

# Custom CSS for that extra polish
css = """
.container { max-width: 900px; margin: auto; padding-top: 2rem; }
.header { text-align: center; margin-bottom: 2rem; }
.vocab-text { color: #64748b; line-height: 1.6; }
"""

with gr.Blocks(title="Neural Audio Classifier") as demo:
    with gr.Column(elem_classes="container"):
        # Header Section
        with gr.Column(elem_classes="header"):
            gr.Markdown(
                """
                # 🎙️ Neural Audio Classifier
                ### Command Recognition System (RNN vs LSTM)
                """
            )
            
        with gr.Row():
            # Left side: Input
            with gr.Column(scale=1):
                with gr.Group():
                    audio_input = gr.Audio(
                        label="Speak or Upload",
                        type="filepath",
                        sources=["microphone", "upload"]
                    )
                    submit_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
                
                with gr.Accordion("📚 See Supported Vocabulary", open=False):
                    gr.Markdown(f"*{vocab}*", elem_classes="vocab-text")

            # Right side: Results
            with gr.Column(scale=1):
                output_label = gr.Label(
                    num_top_classes=5,
                    label="Classification Result"
                )

        submit_btn.click(
            fn=classify_audio,
            inputs=audio_input,
            outputs=output_label
        )
        
        gr.Markdown(
            """
            ---
            <p style='text-align: center;'>Verified at 16kHz mono audio.</p>
            """
        )

if __name__ == "__main__":
    demo.launch(theme=theme, css=css, share=True)
