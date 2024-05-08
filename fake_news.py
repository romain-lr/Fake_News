import numpy as np
import gzip
import pandas as pd

filepath1 = "./train_fake_news.csv"
training_set = pd.read_csv(filepath1)
training_set.columns = ['A', 'B', 'C']
del training_set['A']

def detect_fake_news(title, training_set):
    Cx1 = len(gzip.compress(title.encode()))
    
    distance_from_x1 = []

    for x2 in training_set['C']:
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([title, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)

    sorted_idx = np.argsort(np.array(distance_from_x1))
    top_k_class = training_set.loc[sorted_idx[:10], 'B']
    predict_class = top_k_class.mode().iloc[0]

    return "Fake News" if predict_class == 1 else "Real News"





