from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Initialisation du modèle BERT et du tokenizer pour le traitement du texte
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Passage du modèle en mode évaluation pour désactiver les mécanismes spécifiques à l'entraînement


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fonction pour extraire des embeddings par petits lots
def extract_features(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():  # Désactivation du calcul du gradient pour réduire la consommation de mémoire
          
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
    
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_embeddings.append(embeddings)
    return np.concatenate(all_embeddings, axis=0)


df1['text'] = df1['text'].astype(str)
texts = df1['text'].tolist()

# Extraction des caractéristiques des textes en utilisant la fonction définie plus haut
features = extract_features(texts, batch_size=16)