"""
Codice per trovare le aziende che maggiormente
partecipano ad un dato autovettore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utilities as ut

#leggo le matrice normalizzata calcolata in remove_global_market.py
correps = np.load(r"C:\Users\franc\Documents\magistrale\sistemi complessi\esame_sistemi_complessi\codici\norm_correlation.npy",allow_pickle='TRUE')

#leggo gli indici utilizzati nell'analisi dal file prodotto in correlation.py
index = np.load(r"C:\Users\franc\Documents\magistrale\sistemi complessi\esame_sistemi_complessi\codici\indici.npy",allow_pickle='TRUE')

#leggo il file conetente le informazioni generali dei titoli creato nel file dataset.py
path = r"C:\Users\franc\Documents\magistrale\sistemi complessi\esame_sistemi_complessi\codici\dataset_infromation.csv"
info = pd.read_csv(path)

"""
calcolo autovettori e autovalori
li ordino in modo che il primo autovettore sia
associato all'autovalore più piccolo
"""
eigval, eigvec = np.linalg.eig(correps)
eigvals = np.sort(eigval)
eigvecs = eigvec[:,eigval.argsort()]


N = len(index)

#trovo indice dei massimi (il secondo argomento è il numero dei massimi)
pos = ut.find_max(eigvecs[:,N-2], 10)
"""
creo una lista che abbia come elementi i
titoli corrispondenti alla posizione dei massimi
fondamentalmente è un cambio di base, scrivo l'autovettore
nella base originaria dei titoli
"""
ticker = [index[int(p)] for p in pos]


#sapendo la lista dei titoli maggiori ne trovo le informazioni come nome e settore
find_info(info, ticker)