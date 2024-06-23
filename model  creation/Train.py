from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, df1['target'], test_size=0.1, shuffle=True, random_state=42)

# Création du pipeline avec probability=True pour le modèle SVC
pipeline_SVM = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),  # On évite de retirer la moyenne pour les données sparse, si nécessaire.
    ('svm', SVC(kernel='rbf', probability=True)) 
])


# Configuration des paramètres pour GridSearchCV
parameters_SVM = {'svm__C': [0.7, 0.8, 0.9, 1,5]}

# Configuration de la validation croisée
grid_search_SVM = GridSearchCV(pipeline_SVM,parameters_SVM, cv=3, verbose=1, n_jobs=-1)

# Entrainement
import warnings
warnings.filterwarnings("ignore")

mod_SVM = grid_search_SVM.fit(X_train,y_train)