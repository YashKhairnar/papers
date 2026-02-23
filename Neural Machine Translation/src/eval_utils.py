import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf

def calculate_bleu(target, predicted):
    """
    Calculates the BLEU score between a target sentence and a predicted sentence.
    Both inputs should be lists of tokens.
    """
    smoothing = SmoothingFunction().method1
    return sentence_bleu([target], predicted, smoothing_function=smoothing)

def calculate_chrf(target, predicted):
    """
    Calculates the chrF score (character n-gram F-score).
    Better for morphologically rich languages like Marathi.
    """
    # nltk.sentence_chrf expects strings, not lists of tokens
    return sentence_chrf(" ".join(target), " ".join(predicted))

def translate_sentence(sentence, vocab, model, device, max_len=60):
    """
    Translates a single sentence using the trained multi-layer model.
    """
    model.eval()
    
    # sentence is a list of subword IDs
    # Ensure it has BOS and EOS
    if sentence[0] != vocab.bos_id:
        sentence = [vocab.bos_id] + sentence + [vocab.eos_id]
        
    src_tensor = torch.LongTensor(sentence).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(sentence)])
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    
    mask = (src_tensor == 0).to(device)
    trg_indices = [vocab.bos_id]
    attentions = []
    
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask=mask)
            
        attentions.append(attention)
        prediction = output.argmax(1).item()
        
        if prediction == vocab.eos_id:
            break
        trg_indices.append(prediction)
            
    # Convert indices back to pieces/words
    prediction_pieces = [vocab.itos[idx] for idx in trg_indices[1:]]
    attentions = torch.cat(attentions, dim=0).squeeze(1) if attentions else torch.tensor([])
    
    return prediction_pieces, attentions

def run_test_evaluation(model, dataset, device, num_samples=None):
    """
    Evaluates the model on the provided dataset split (e.g., test set).
    Calculates aggregate BLEU and chrF scores.
    """
    model.eval()
    total_bleu = 0
    total_chrf = 0
    samples = []
    
    # If num_samples is None, evaluate the whole set
    eval_range = range(len(dataset)) if num_samples is None else range(min(num_samples, len(dataset)))
    
    print(f"Starting evaluation on {len(eval_range)} samples...")
    for i in eval_range:
        if i % 10 == 0:
            print(f"Processing sample {i}/{len(eval_range)}...")
        
        src_tensor, trg_tensor = dataset[i]
        src_ids = src_tensor.tolist()
        trg_ids = trg_tensor.tolist()
        
        # Translate
        translated_pieces, _ = translate_sentence(src_ids, dataset.vocab, model, device)
        
        # Reference (ignore BOS/EOS/PAD)
        reference_pieces = [dataset.vocab.itos[idx] for idx in trg_ids if idx not in [dataset.vocab.pad_id, dataset.vocab.bos_id, dataset.vocab.eos_id]]
        
        # Scores
        total_bleu += calculate_bleu(reference_pieces, translated_pieces)
        total_chrf += calculate_chrf(reference_pieces, translated_pieces)
        
        # Save a few examples for qualitative review
        if i < 10:
            samples.append({
                'src': dataset.vocab.decode(src_ids),
                'ref': dataset.vocab.decode(trg_ids),
                'pred': " ".join(translated_pieces).replace(" ", "").replace(" ", " ")
            })
            
    avg_bleu = (total_bleu / len(eval_range)) * 100
    avg_chrf = (total_chrf / len(eval_range)) * 100
    
    print(f"\n--- Evaluation Results ---")
    print(f"Average BLEU Score: {avg_bleu:.2f}")
    print(f"Average chrF Score: {avg_chrf:.2f}")
    
    return avg_bleu, avg_chrf, samples

