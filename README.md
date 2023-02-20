# CVDefectClassification
CV Challange


Interessanti:
Rimozione pattern -> https://stackoverflow.com/questions/52108147/how-can-i-remove-the-back-decorative-pattern-for-my-ocr
https://arxiv.org/pdf/1807.02894.pdf
https://www.mdpi.com/1424-8220/21/13/4292


spunti:
individuare gli outlayer
normalizzare le immagini lavorando su brillantezza, alcune troppo scure.
silency measure per trovare rotture.
Provare gradiente per riconoscere pattern
smoothness

detect shape(small circles) and remove them

estrarre n immagini diverse con filtri/altro e passarle in input ad una ccn (come se l'input layer fosse un layer di feature maps)



Tentativi fatti
thresholding -> global, adaptive, otus (Non buoni risultati in generale, ok su immagini pulite ma aggiunge rumore a quelle scure)
normalizzazione immagine -> buoni risultati
calcolo del gradiente con filtro laplaciano -> *2 perchè troppo anonimo
gradiente sottratto a immagini per evidenziare problemi
operazioni morfologiche -> opening, closing, dilatazioni e erosioni


analisi concetrata sul gradiente:
    incremento di 10 volte -> notiamo molto rumore sale e pepe
    applicazione gradiente ad immagini non std -> rumore meno evidente, forse è più facile applicare i filtri a queste immagini piuttosto che a quelle standardizzate
    implementazione filtro per rimozione sale e pepe
    ottenuta buona sequenza con: std, grad, blur e sottrazione all'immagine
    provo il threshold -> non ottengo risultati particolarmente discriminativi, va provato più approfonditamente

analisi su opening:
    sembra filtrare meglio opening con filtri circolare (11, 11)



useful:
RayTune -> https://h-huang.github.io/tutorials/beginner/hyperparameter_tuning_tutorial.html