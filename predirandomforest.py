# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:12:37 2023

@author: achil
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load

def predict_fake_news(title):
    # Charger le modèle sauvegardé
    modelpath = r"./random_forest_model.joblib"
    model = load(modelpath)

    # Charger les données d'entraînement depuis le fichier CSV (pour ajuster le vectorizer)
    train_data = pd.read_csv(r"./train_fake_news.csv")
    X_train = train_data['text']

    # Charger le vectorizer (ou créer un nouveau si nécessaire)
    vectorizer = TfidfVectorizer()

    # Ajuster le vectorizer avec les données d'entraînement
    X_train_tf_idf = vectorizer.fit_transform(X_train)

    # Transformer le titre avec le vectorizer ajusté
    title_tf_idf = vectorizer.transform([title])

    # Prédire si le titre est une fake news ou non
    prediction = model.predict(title_tf_idf)

    # Retourner le résultat
    return "Fake News" if prediction[0] == 1 else "Real News"
