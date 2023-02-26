from data.preprocessing import ELImgPreprocessing
from contextlib import redirect_stdout
from datetime import datetime
from keras.callbacks import EarlyStopping
from common_utils import *
import random as rnd

WORKING = 0
NOT_WORKING = 1


def data_augmentation(sample):
    prob_ud = rnd.random()
    prob_lr = rnd.random()
    if prob_ud >= 0.5:
        sample = np.flipud(sample)
    if prob_lr >= 0.5:
        sample = np.fliplr(sample)
    return sample


def dataset_augmentation(_dataset):
    print("Augmenting dataset")
    new_dataset = []
    for sample in _dataset:
        new_dataset.append(data_augmentation(sample))
    return np.array(new_dataset)


def save_model(_model, _model_folder, idx):
    _model.save_weights(os.path.join(_model_folder, f'model_{idx}.h5'))
    with open(os.path.join(_model_folder, f'modelsummary_{idx}.txt'), 'w') as f:
        with redirect_stdout(f):
            _model.summary()


def fit_and_save(_model, _x_train, _y_train):
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
    _model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        class_weight={0: 1, 1: 2.5},
        epochs=20,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    save_model(_model, model_folder, i)
    return _model


if __name__ == '__main__':
    if PREPROCESS:
        preprocessing = ELImgPreprocessing()
        preprocessing.preprocess()
    dataset = np.load(os.path.join(DATA_PATH_PROCESSED, "processed_data.npy"), allow_pickle=True)
    train_folds = get_folds(TRAIN_PATH)
    test_folds = get_folds(TEST_PATH)

    model_folder = os.path.join('models/', datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)"))
    os.makedirs(model_folder, exist_ok=True)
    analytics_table = dict()
    for i in range(10):
        analytics_table[str(i)] = dict()
        print(f'Iteration: {i}')
        x_train, y_train = split_samples_labels(dataset[train_folds[i].astype(np.int64)])
        x_test, y_test = split_samples_labels(dataset[test_folds[i].astype(np.int64)])

        x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, -1)

        model = get_model()
        fit_and_save(model, x_train, y_train)

        y_pred = model.predict(x_test)
        y_pred = y_pred > 0.5

        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        analytics_table[str(i)]['f1'] = f1
        analytics_table[str(i)]['accuracy'] = accuracy

        print(f'F1 score: {f1}')
        print(f'Accuracy: {accuracy}')
        print("-----------------\n\n")

    all_accuracy = [analytics_table[key]['accuracy'] for key in analytics_table]
    all_f1 = [analytics_table[key]['f1'] for key in analytics_table]
    pd.DataFrame(analytics_table).to_csv(os.path.join(model_folder, 'analytics_table.csv'))
    sum_up_table = calculate_sum_up_table(analytics_table)
    pd.DataFrame(sum_up_table, index=[0]).to_csv(os.path.join(model_folder, 'sum_up_table.csv'))

    print('END')
