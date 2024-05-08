import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

filepath = r"./train_fake_news.csv"

# Charger les données d'entraînement depuis le fichier CSV
train_data = pd.read_csv(filepath)

# Diviser les données d'entraînement avec la valeur Vraie ou Fausse (y_train) du titre de l'article (X_train)
X_train = train_data['text']
y_train = train_data['label']

# On vectorise les titres des articles, des strings, avec le protocole TF-IDF pour qu'il soit interprété lors des arbres de décision
vectorizer = TfidfVectorizer()
X_train_tf_idf = vectorizer.fit_transform(X_train)
# print(X_train) - Affichage des X_train pour vérifier le bon fonctionnement du code
# print(X_train_tf_idf) - Affichage des X_train vectorisé pour vérifier le bon fonctionnement du code
# Création du modèle Random Forest, avec un nombre de 200 arbres de décision différent
model = RandomForestClassifier(n_estimators=200) 

# Entraînement du modèle sur les données train
model.fit(X_train_tf_idf, y_train)

# Sauvegarde du modèle dans un fichier pour une utilisation ultérieure
from joblib import dump
dump(model, 'random_forest_model.joblib')

"""
Test pour savoir le modèle est bien créé, qu'il marche etc...

# Prédiction sur les données de test
y_pred = model.predict(X_train_tf_idf)

# Évaluation de la performance du modèle
accuracy = accuracy_score(y_train, y_pred)  # Pour la classification, utilisez d'autres métriques pour la régression
print(f'Accuracy: {accuracy}')
"""