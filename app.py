import streamlit as st
import joblib
import torch
import re
from transformers import BertTokenizer, BertModel

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

# Interface utilisateur avec Streamlit
st.title("Text Classification with BERT and SVM")

text_input = st.text_area("Enter text for classification:")

if st.button("Classify"):
    cleaned_text = clean_text(text_input)
    features = extract_features([cleaned_text])
    prediction = svm_model.predict(features)
    st.write(f"Prediction: {prediction[0]}")
