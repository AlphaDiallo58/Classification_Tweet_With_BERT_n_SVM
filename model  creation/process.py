# Fonction pour le nettoyage de la colonne 'text'
import re
def clean_text(text):
    text = text.lower()  # Conversion en minuscules
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Supprime les URLs
    text = re.sub(r'@\w+', '', text)  # Supprime les mentions
    text = re.sub(r'#', '', text)  # Supprime les hashtags
    text = re.sub(r'\n', ' ', text)  # Remplace les sauts de ligne par des espaces
    text = re.sub(r'[^a-z\s]', '', text)  # Supprime tous les caractères non alphabétiques
    text = re.sub(r'\s+', ' ', text).strip()  # Supprime les espaces supplémentaires
    return text


df1['text'] = df1['text'].apply(clean_text)