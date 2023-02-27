<img width="1000" alt="Screenshot 2023-02-27 at 11 45 18" src="https://user-images.githubusercontent.com/52169960/221543307-ca30d431-a1a3-4f28-a18e-dbd7c4180ed7.png">



## Project Structure
In this project you will find several folders and files useful for code execution. <br>
This project has 3 main scripts:
- ***preprocess.py*** (used for data preprocessing and creation of npy file)
- ***train.py*** (used to train the model, uses the output of preprocess.py)
- ***test.py*** (used to execute only the test phase on 10 folds)

The project also have several folders:
- ***best_model_metrics*** (contains analytics table and sum up table of the best model)
- ***best_model_weights*** (contains '.h5' files used to load model weights in test phase)
- ***boostrap_folds*** (contains train and test folds info)
- ***data*** (contains the dataset and the output of the preprocess.py script)
- ***models*** (in this folder are stored the models trained using the train.py script)

## <i><u>Analytics</u></i> and <i><u>Sum-up</u></i> tables
You can find these tables (referring to the best model) inside the 'best_model_metrics' folder in the root of this project. <br>
### Analytics table
The Analytics table contains f1-score and accuracy-score for each fold of the best model.
### Sum-Up table
The Sum-Up table contains the mean values for both f1 and accuracy. <br>
This table also contains the standard deviation for the accuracy score.


## Execution instructions
In this project you can decide if you want to perform only the preprocessing, the training or the test of the model.

### Virtual Environment
We suggest you to create a dedicated virtual environment before starting the execution.
If you don't want to create it then just skip this section.

To create the virtual environment follow these steps:
1. Go to the project's root directory
2. Create the virtual environment with the following command:
    > python3 -m venv venv/

3. Activate it:
    > source venv/bin/activate

### Installing dependencies
To install all the needed dependencies you just need to run:
> pip install -r requirements.txt


### Preprocess
To run the preprocessing on the data you can easily run the following command in the root of the project
> python3 preprocess.py

### Train
To train a new model using the given 10 folds you can easily run the following command in the root of the project
> python3 train.py

N.B. Remember to run the preprocess.py before the train.py script if the file data > processed > processed_data.npy is missing.

### Test
If you want to perform only the test of the model you can simply execute the command:
> python3 test.py

From the root of the project.<br><br>
Be aware that best_model_weights folder must contain an '.h5' for each fold that has to be tested (the index value of the model_{index}.h5 file is equal to fold_number - 1).
