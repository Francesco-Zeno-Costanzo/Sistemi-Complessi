"""
Codice per il calcolo dell' inverse participation ratio
"""

import numpy as np
import matplotlib.pyplot as plt

import utilities as ut

#leggo una matrice calcolata nel file Correlation.py
crosscor = np.load(r"C:\Users\franc\Documents\magistrale\sistemi complessi\esame_sistemi_complessi\codici\cross_correlation.npy",allow_pickle='TRUE')

"""
calcolo autovettori e autovalori
li ordino in modo che il primo autovettore sia
associato all'autovalore più piccolo
"""
eigval, eigvec = np.linalg.eig(crosscor)
eigvals1 = np.sort(eigval)
eigvecs1 = eigvec[:,eigval.argsort()]

N = len(eigvals1)
L = 1259


I0 = ut.I(N, eigvecs1)
print(f'numero partecipanti ultimo autovettore: {1/I0[-1]}')
print(f'numero partecipanti primo autovettore: {1/I0[0]}')

A = np.random.randn(N,L)
C = np.dot(A, A.T)/L

q = L/N
l1 = 1+1/q -2*np.sqrt(1/q)
l2 = 1+1/q +2*np.sqrt(1/q)

eigval, eigvec = np.linalg.eig(C)
eigvals2 = np.sort(eigval)
eigvecs2 = eigvec[:,eigval.argsort()]
I1 = I(N, eigvecs2)


fig, ax = plt.subplots()
plt.title('Inverse participation ratio', fontsize=15)
plt.ylabel('IPR($\lambda$)', fontsize=15)
plt.xlabel('$\lambda$', fontsize=15)
plt.plot(eigvals1, I0, marker='.', linestyle='', label='mercato')
plt.plot(eigvals2, I1, marker='.', linestyle='', label='mat_ran')
ax.fill_between(eigvals2, 0.001, 0.4, color='green', alpha=0.5)
plt.grid()
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')
plt.show()