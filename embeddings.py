from transformers import BertTokenizer, BertModel
import torch
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
nltk.download('punkt_tab')
  

def load_cleaned_dreams(file_path: str) -> dict:
    with open("/Users/work/Desktop/Projects/Eerie/cleaned_ten_nights_of_dreams.json", "r", encoding="utf-8") as f:
        return json.load(f)

def generate_bert_embeddings(text: str, tokenizer, model) -> torch.Tensor:

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding

def generate_dream_embeddings(dreams: dict, tokenizer, model) -> dict:
    print(f"Total number of dreams: {len(dreams)}")
    for header, content in dreams.items():
        print(f"Night: {header}, Content Length: {len(content)}")
    embeddings = {}
    for header, content in dreams.items():
        print(f"Generating embedding for {header}...")
        emb = generate_bert_embeddings(content, tokenizer, model)
        embeddings[header] = emb.detach().numpy()
        print(f"Embedding for {header} shape: {emb.shape}")
    return embeddings

def analyze_sentiment(dreams: dict):

    analyzer = SentimentIntensityAnalyzer()
    analysis_results = {}

    for header, content in dreams.items():
        sentiment_score = analyzer.polarity_scores(content)['compound']
        keywords = TextBlob(content).noun_phrases
        analysis_results[header] = {
            "sentiment_score": sentiment_score,
            "keywords": keywords
        }

    return analysis_results

if __name__ == "__main__":
    cleaned_dreams_file = "cleaned_ten_nights_of_dreams.json"
    
    cleaned_dreams = load_cleaned_dreams(cleaned_dreams_file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    sentiment_results = analyze_sentiment(cleaned_dreams)

    for header, result in sentiment_results.items():
        print(f"{header} - Sentiment: {result['sentiment_score']} - Keywords: {result['keywords'][:5]}")
    dream_embeddings = generate_dream_embeddings(cleaned_dreams, tokenizer, model)
    output_embeddings_file = "dream_embeddings.json"
    with open(output_embeddings_file, "w", encoding="utf-8") as f:
        json.dump({header: emb.tolist() for header, emb in dream_embeddings.items()}, f, indent=4)
    
    print(f"Dream embeddings saved to {output_embeddings_file}.")

    sentiment_file  = "sentiments.json"
    with open(sentiment_file, "w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, indent=4)
    
    print(f"Semantics saved to {sentiment_file}.")