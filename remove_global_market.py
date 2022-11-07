"""
Codice per rimuovere l'andamento globale del mercato
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import utilities as ut

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
L = 1259

M = ut.pro(retunorm, eigvecs, N-1, L, N)


def f(x, b, c):
    """
    modello per fit
    """
    return b*x + c

beta = np.zeros(N)
alpha = np.zeros(N)
#plt.figure(8)
#sottraggo andamento del mercato
for i in range(N):
    x = M
    y = retunorm[i,:]
    popt, pcov = curve_fit(f, x, y)
    beta[i] = popt[0]
    #if i > 480:
    #    t = np.linspace(np.min(x), np.max(x), 1000)
    #    plt.plot(x, y, marker='.', linestyle='')
    #    plt.plot(t, f(t, *popt), linestyle='-')
    alpha[i] = np.mean(y) - beta[i]*np.mean(x)

epsilon = np.zeros((N, L))
correps = np.zeros((N, N))

for i in range(N):
    epsilon[i, :] = retunorm[i,:] - alpha[i] - beta[i]*M


for i in range(N):
    for j in range(N):
        correps[i, j] = ut.corr(epsilon[i, :], epsilon[j, :])


#prob dens func nel limite, N, L infiniti
P = ut.dens_prob(L, N)
sup = P['supporto']
pdf = P['pdf']


#salvo la matrice rinormalizzata
np.save("norm_correlation.npy", correps)

plt.figure(1)
x = np.reshape(correps, N*N)
z = np.reshape(crosscor, N*N)
plt.title('Distribuzione dei coefficenti della cross correlation',fontsize=15)
plt.xlabel('cross correlation $C_{ij}$',fontsize=15)
plt.ylabel('P($C_{ij}$)',fontsize=15)
plt.yscale('log')
plt.grid()
plt.hist(x, int(np.sqrt(N*N-1)), histtype='step', label='$Cr_{ij}$')
plt.hist(z, int(np.sqrt(N*N-1)), histtype='step', label='$C_{ij}$')
plt.legend(loc='best')


plt.figure(2)
y = np.linalg.eigvalsh(correps)
plt.title('Distribuzione degli autovalori di $Cr_{ij}$',fontsize=15)
plt.xlabel('autovalori $\lambda$ di $Cr_{ij}$', fontsize=15)
plt.ylabel('P($\lambda$)', fontsize=15)
plt.grid()
plt.yscale('log')
plt.hist(y, N, density=True)
plt.plot(sup, pdf, 'k')

plt.show()