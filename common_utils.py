import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import keras.backend as K

TEST_PATH = 'bootstrap_folds/test_folds/'
TRAIN_PATH = 'bootstrap_folds/train_folds/'
DATA_PATH_PROCESSED = 'data/processed/'


def get_sorted_dict(unsortable_names):
    sortable_dict = dict()
    for name in unsortable_names:
        first_part = name.split('.')[0]
        number = first_part.split('_')[2]
        filled = number.zfill(2)
        sortable_dict[filled] = name
    return [sortable_dict[key] for key in sorted(sortable_dict)]


def get_folds(path):
    print('Reading folds indexes')
    fold_files = os.listdir(path)
    fold_files = get_sorted_dict(fold_files)
    folds = []
    for file_name in fold_files:
        print(f'Loading fold: {file_name}')
        fold = pd.read_csv(os.path.join(path, file_name), index_col=0, delimiter=',')
        folds.append(np.array(fold[fold.columns[0]].tolist()))
    return np.array(folds, dtype=object)


def split_samples_labels(_dataset):
    return np.array([x for x in _dataset[:, 0]]), np.array([y for y in _dataset[:, 1]])


def calculate_f1(y_true, y_pred):
    print('Calculating f1 score')
    f1 = f1_score(y_true, y_pred)
    return round(f1, 2)


def calculate_accuracy(y_true, y_pred):
    print('Calculating accuracy score')
    accuracy = accuracy_score(y_true, y_pred)
    return round(accuracy, 2)


def calculate_sum_up_table(analytics_table):
    print('Calculating averages metrics')
    all_accuracy = [analytics_table[key]['accuracy'] for key in analytics_table]
    all_f1 = [analytics_table[key]['f1'] for key in analytics_table]
    sum_up_table = dict()
    sum_up_table['f1'] = np.round(np.mean(all_f1), 2)
    sum_up_table['accuracy'] = np.round(np.mean(all_accuracy), 2)
    sum_up_table['std_accuracy'] = np.round(np.std(all_accuracy), 2)
    return sum_up_table


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
