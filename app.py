from flask import Flask, request, render_template
import joblib
import torch
import re
from transformers import BertTokenizer, BertModel
import numpy as np 
import pandas as pd

app = Flask(__name__)

# Charger le modèle SVM
svm_model = joblib.load('svm_model.pkl')

# Initialisation du tokenizer et du modèle BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

# Fonction de nettoyage de texte
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fonction pour extraire des embeddings par petits lots
def extract_features(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_embeddings.append(embeddings)
    return np.concatenate(all_embeddings, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = clean_text(text)
        features = extract_features([cleaned_text])
        prediction = svm_model.predict(features)[0]
        result = 'Positive' if prediction == 0 else 'Negative'
        return render_template('index.html', prediction=result, text=text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
