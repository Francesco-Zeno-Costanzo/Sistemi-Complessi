"""
Codice per vedere il comportamento
delle componenti degli autovettori
"""

import numpy as np
import matplotlib.pyplot as plt

crosscor = np.load(r"C:\Users\franc\Documents\magistrale\sistemi complessi\esame_sistemi_complessi\codici\cross_correlation.npy",allow_pickle='TRUE')


#calcolo autovettori e autovalori
eigval, eigvec = np.linalg.eig(crosscor)

"""
li ordino in modo che il primo autovettore sia
associato all'autovalore più piccolo
"""
eigvals = np.sort(eigval)
eigvecs = eigvec[:,eigval.argsort()]
N = len(eigvals)

#valori da plottare
val = [220, N-3, N-2, N-1]
plt.figure(1)
plt.suptitle('Componenti autovettori', fontsize=15)

for i, l in enumerate(val):
    vec = eigvecs[:,l]/(np.sqrt(np.mean(eigvecs[:,l]**2)-np.mean(eigvecs[:,l])**2))
    x = np.linspace(-8, 8, 10000)
    plt.subplot(2,2,i+1)
    plt.title(f'autovettore numero {l+1} \n relativo a $\lambda$={eigvals[l]:.2f}')
    plt.hist(vec, int(np.sqrt(N-1))-7, density=True)
    plt.grid()
    plt.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi))
    plt.xlim(-8, 8)

plt.show()