"""
Codice per verificare la presenza di un andamento
globale che si ripercuote sui titoli
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import utilities as ut

#inizio e fine della lettura degli storici
start_story = '2017-10-03'
end_story = '2022-10-03'

#Leggo lo storico dell'indice totale
SP500 = yf.download('^GSPC', start=start_story, end=end_story, interval='1d', progress=False)
L = len(SP500)

#leggo le matrici calcolate nel file Correlation.py
crosscor = np.load(r"C:\Users\franc\Documents\magistrale\sistemi complessi\esame_sistemi_complessi\codici\cross_correlation.npy",allow_pickle='TRUE')
retunorm = np.load(r"C:\Users\franc\Documents\magistrale\sistemi complessi\esame_sistemi_complessi\codici\normalized_return.npy",allow_pickle='TRUE')

"""
calcolo autovettori e autovalori
li ordino in modo che il primo autovettore sia
associato all'autovalore più piccolo
"""
eigval, eigvec = np.linalg.eig(crosscor)
eigvals = np.sort(eigval)
eigvecs = eigvec[:,eigval.argsort()]

N = len(eigvals)


#calcolo del ritorno ad un giorno normalizzato del SP500
g = np.log(SP500['Close']) - np.log(SP500['Open'])
sigma = np.sqrt(np.mean(g**2)-np.mean(g)**2)
g1 = (g - np.mean(g))/sigma

GN = ut.pro(retunorm, eigvecs, N-1, L, N)
G220 = ut.pro(retunorm, eigvecs, 220, L, N)

CN = ut.corr(g1, GN)
C220 = ut.corr(g1, G220)



print(f'Correlazione con ultimo autovalore: {CN:.3f}')
print(f'Correlazione con 220-esimo autovalore: {C220:.3f}')

plt.figure(1)
plt.subplot(121)
plt.ylabel("ritorni normalizzati proiettati sull'ultimo autovettore")
plt.xlabel("ritorni normalizzati dello SP500")
plt.plot(g1, GN, marker='.', linestyle='')
plt.grid()

plt.subplot(122)
plt.ylabel("ritorni normalizzati proiettati 220-esimo autovettore")
plt.xlabel("ritorni normalizzati dello SP500")
plt.axis([-6, 6, -6, 6])
plt.plot(g1, G220, marker='.', linestyle='')
plt.grid()

plt.show()