"""
Codice per creazione vera matrice random
"""

import numpy as np
import matplotlib.pyplot as plt

from utilities import dens_prob

#Dimensioni matrice
L = 1259
N = 482

#matrice con entrate gaussiane con \mu = 0 e \sigma = 1
A = np.random.randn(N,L)
C = np.dot(A, A.T)/L

#calcolo autovalori
y = np.linalg.eigvalsh(C)
plt.hist(y, int(np.sqrt(N-1)), density=True)

#prob dens func nel limite, N, L infiniti
P = dens_prob(L, N)
sup = P['supporto']
pdf = P['pdf']


plt.figure(1)
plt.title('Distribuzione autovalori', fontsize=15)
plt.ylabel('P($\lambda$)', fontsize=15)
plt.xlabel('$\lambda$', fontsize=15)
plt.grid()
plt.plot(sup, pdf, 'k')

#calcolo autovettori e autovalori
eigval, eigvec = np.linalg.eig(C)
eigvals = np.sort(eigval)
eigvecs = eigvec[:,eigval.argsort()]
N = len(eigvals)

#valori da plottare
val = [i*(N-1)//4 for i in range(1, 5)]
plt.figure(2)
plt.suptitle('Componenti autovettori', fontsize=15)

for i, l in enumerate(val):
    #normalizzo autovettore
    vec = eigvecs[:,l]/(np.sqrt(np.mean(eigvecs[:,l]**2)-np.mean(eigvecs[:,l])**2))
    x = np.linspace(-8, 8, 10000)

    plt.subplot(2,2,i+1)
    plt.title(f'autovettore numero {l+1} \n relativo a $\lambda$={eigvals[l]:.2f}')
    plt.hist(vec, int(np.sqrt(N-1))-6, density=True) #vettore
    plt.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), 'k') #distribuzione teorica
    plt.xlim(-8, 8)
    plt.grid()

plt.show()