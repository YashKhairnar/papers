import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def generate_wordclouds(csv_path):
    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if 'src' not in df.columns or 'tgt' not in df.columns:
        print("Error: CSV must contain 'src' and 'tgt' columns.")
        return

    print("Processing English text...")
    english_text = " ".join(str(text) for text in df['src'].dropna())
    
    print("Processing Marathi text...")
    marathi_text = " ".join(str(text) for text in df['tgt'].dropna())

    # Configure high-res wordclouds
    # Note: Marathi requires a font that supports Devanagari script.
    # We will use Arial Unicode MS on Mac, or fallback to default if not found.
    font_path = '/Library/Fonts/Arial Unicode.ttf'
    if not os.path.exists(font_path):
        font_path = None # Fallback to default, though Marathi might render as boxes

    print("Generating English wordcloud...")
    wc_eng = WordCloud(width=1600, height=800, background_color='#f8fafc', colormap='Blues', max_words=1000).generate(english_text)
    
    print("Generating Marathi wordcloud...")
    wc_mar = WordCloud(width=1600, height=800, background_color='#f8fafc', colormap='Wistia', font_path=font_path, max_words=1000).generate(marathi_text)

    # Save English
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wc_eng, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    os.makedirs('../assets', exist_ok=True)
    plt.savefig('../assets/english_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()


    # Save Marathi
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wc_mar, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('../assets/marathi_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Success! Generated '../assets/english_wordcloud.png' and '../assets/marathi_wordcloud.png'.")


if __name__ == "__main__":
    generate_wordclouds("../data/samantar_dataset.csv")
