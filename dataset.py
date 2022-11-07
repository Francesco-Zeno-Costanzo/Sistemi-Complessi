"""
Codice per la creazione del dataset da analizzare
"""

import time
import numpy as np
import datapackage
import pandas as pd
import yfinance as yf

start = time.time()
#inizio e fine della lettura degli storici
start_story = '2017-10-03'
end_story = '2022-10-03'

#Data set contenente informazioni sui titoli dello S&P500
DATA_URL = r"https://datahub.io/core/s-and-p-500-companies/datapackage.json"
package = datapackage.Package(DATA_URL)
resources = package.resources
'''
interessano solo i dati scritti in tabella che
contengono: simbolo, nome e settore dei titoli
'''
for resource in resources:
    if resource.tabular:
        dataset = pd.read_csv(resource.descriptor['path'])

'''
ora dataset contiene: simbolo, nome e settore dei titoli
lo riordino in onrdine alfabetico del simbolo e riaggiusto
l'indice che lo labella
'''
dataset = dataset.sort_values('Symbol').reset_index(drop=True)
#salvo le infomezioni su un csv
dataset.to_csv('dataset_infromation.csv', index=False)

#histoty è un dizionario che conterrà gli storici dei vari titoli
history = {}
dataset_len = len(dataset)

for index, row in dataset.iterrows():
    #ciclo su tutto il dataset leggendone il simbolo
    ticker = row['Symbol']
    print(f"[{(index + 1)}/{dataset_len}] Scarico dati di: {ticker}")

    #scarico la storia del titolo, in un certo range con un certo intervallo
    hist = yf.download(ticker, start=start_story, end=end_story, interval='1d', progress=False)

    #controllo che si sia potuto scaricare tutto
    if hist.empty:
        print(f"La storia di: {ticker} non è disponibile, passo al prossimo titolo")
    else:
        history[ticker] = hist

"""
Salvo in locale le storie per non leggerle ogni volta.
Usare estensione .npy velocizza la lattura
"""
np.save("dataset.npy", history)

mins = (time.time()-start)//60
sec = (time.time()-start) % 60

print(f"Tempo impiegato: {mins} minuniti {sec:.2f} secondi")