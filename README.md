# CVDefectClassification
### Powered By Simone Pio Caronia & Paolo Ruggirello

## Project Structure
In this project you will find several folders and files useful for code execution. <br>
This project has 3 main scripts:
- preprocess.py (used for data preprocessing)
- train.py (used to train the cv model, uses the output of process.py)
- test.py (used to execute only the test phase on 10 folds)

The project also have several folders:
- best_model_metrics (contains analytics table and sum up table of the best model)
- best_model_weights (contains '.h5' files used to load model weights in test phase)
- boostrap_folds (contains train and test folds info)
- data (contains the dataset and the output of the preprocess.py script)
- models (in this folder are stored the models trained using the train.py script)

## 'Analytics' and 'Sum-up' tables
You can find these tables (referring to the best model) inside the 'best_model_metrics' folder in the root of this project. <br>
### Analytics table
The Analytics table contains f1-score and accuracy-score for each fold of the best model.
### Sum-Up table
The Sum-Up table contains the mean values for both f1 and accuracy. <br>
This table also contains the standard deviation for the accuracy score.


## Execution instructions
In this project you can decide if you want to perform only the preprocessing, the training or the test of the model.

### Build virtual environment
Remember to create a virtual environment before running the code. To do that you can follow these instructions:


### Prepprocessing
To run the preprocessing you can easly run the following command in the root of the project
>




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


Possible implementation:
optimizer: adam -> https://www.tensorflow.org/api_docs/python/tf/keras/optimizers