"""
Codice per il calcolo delle correlazioni
"""

import time
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import utilities as ut

start = time.time()

#inizio e fine della lettura degli storici
start_story = '2017-10-03'
end_story = '2022-10-03'

#Leggo lo storico dell'indice totale
print('Leggo storico SP500')
SP500 = yf.download('^GSPC', start=start_story, end=end_story, interval='1d', progress=False)
L = len(SP500)

#leggo gli storici che erano stati scaricati
history = np.load(r"C:\Users\franc\Documents\magistrale\sistemi complessi\esame_sistemi_complessi\codici\dataset.npy",allow_pickle='TRUE').item()

"""
Alcuni titoli potrebbero essere 'nati' durante il periodo
di tempo considerato e vengono quindi esclusi dall'analisi
"""
index = [] #lista che conterrà i titoli da analizzare

for i, ticker in enumerate(history.keys()):
    #prendo l'apertura del singolo indice
    apertura = history[ticker]['Open']
    """
    se la storia è di lunghezza diversa da quella dell SP500
    (in linea di pricipio può essere solo più corta) escludo
    il titolo dall'analisi
    """
    if len(apertura) != L:
        print(f'Il titolo {ticker} ha una storia più corta, verrà scartato')
    else:
        index.append(ticker) #conserovo i titoli buoni


N = len(index)

print(f'Numero indici: {N}, lunghezza intervallo temporale: {L}')
#marici che conterrano le informazioni importanti
apertura = np.zeros((N, L))
chiusura = np.zeros((N, L))
return1d = np.zeros((N, L))
retunorm = np.zeros((N, L))
crosscor = np.zeros((N, N))

#matrci per lo shuffle
retshuff = np.zeros((N, L))
rtnshuff = np.zeros((N, L))
cscshuff = np.zeros((N, N))



for i, ticker in enumerate(index):

    #leggo i dati e calcolo i ritorni ad un giorno normalizzati
    apertura[i, :], chiusura[i,:] = history[ticker]['Open'], history[ticker]['Close']
    return1d[i, :] = np.log(chiusura[i,:]) - np.log(apertura[i,:])
    sigma = np.sqrt(np.mean(return1d[i, :]**2)-np.mean(return1d[i, :])**2)
    retunorm[i, :] = (return1d[i, :] - np.mean(return1d[i, :]))/sigma

    #come sopra ma opero un mescolamento per distruggere le correlazioni
    retshuff[i, :] = np.random.choice(return1d[i, :], size=L, replace=False)
    sigma = np.sqrt(np.mean(retshuff[i, :]**2)-np.mean(retshuff[i, :])**2)
    rtnshuff[i, :] = (retshuff[i, :] - np.mean(retshuff[i, :]))/sigma



#calcolo delle matrici di correlazione
for i in range(N):
    for j in range(N):
        crosscor[i, j] = ut.corr(retunorm[i, :], retunorm[j, :])
        cscshuff[i, j] = ut.corr(rtnshuff[i, :], rtnshuff[j, :])


#prob dens func nel limite, N, L infiniti
P = ut.dens_prob(L, N)
sup = P['supporto']
pdf = P['pdf']


#salvo su file.npy perché più veloce
np.save("cross_correlation.npy", crosscor)
np.save("normalized_return.npy", retunorm)
np.save("indici.npy", index)

mins = (time.time()-start)//60
sec = (time.time()-start) % 60

print(f"Tempo impiegato: {mins} minuniti {sec:.2f} secondi")

##Grafici correlazioni e autovalori

plt.figure(1)
x = np.reshape(crosscor, N*N)
plt.title('Distribuzione dei coefficenti della cross correlation',fontsize=15)
plt.xlabel('cross correlation $C_{ij}$',fontsize=15)
plt.ylabel('P($C_{ij}$)',fontsize=15)
plt.yscale('log')
plt.grid()
plt.hist(x, int(np.sqrt(N*N-1)))


plt.figure(2)
y = np.linalg.eigvalsh(crosscor)
plt.title('Distribuzione degli autovalori di $C_{ij}$',fontsize=15)
plt.xlabel('autovalori $\lambda$ di $C_{ij}$', fontsize=15)
plt.ylabel('P($\lambda$)', fontsize=15)
plt.grid()
plt.yscale('log')
plt.plot(sup, pdf, 'k')
plt.hist(y, N, density=True)


plt.figure(3)
x = np.reshape(cscshuff, N*N)
plt.title('Distribuzione dei coefficenti della cross correlation\n dopo lo shuffle',fontsize=15)
plt.xlabel('cross correlation $C_{ij}$',fontsize=15)
plt.ylabel('P($C_{ij}$)',fontsize=15)
plt.hist(x, int(np.sqrt(N*N-1)))
plt.yscale('log')
plt.grid()


plt.figure(4)
y = np.linalg.eigvalsh(cscshuff)
plt.title('Distribuzione degli autovalori di $C_{ij}$ dopo lo shuffle',fontsize=15)
plt.xlabel('autovalori $\lambda$ di $C_{ij}$', fontsize=15)
plt.ylabel('P($\lambda$)', fontsize=15)
plt.hist(y, int(np.sqrt(N-1)), density=True)
plt.plot(sup, pdf, 'k')
plt.grid()

plt.show()