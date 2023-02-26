
from keras.applications import DenseNet121

from data.preprocessing import ELImgPreprocessing
from contextlib import redirect_stdout
from datetime import datetime
from keras.callbacks import EarlyStopping
from common_utils import *

# 0 -> cell works
# 1 -> cell doesn't work

WORKING = 0
NOT_WORKING = 1

IN_EXAM_W = ['images/cell0004.png', 'images/cell0067.png', 'images/cell0106.png']
IN_EXAM_NOT_W = ['images/cell0165.png', 'images/cell0220.png', 'images/cell0001.png', 'images/cell0002.png', 'images/cell0057.png']

PREPROCESS = False

densenet = DenseNet121(
    weights='imagenet',  # /kaggle/input/densenet-keras/DenseNet-BC-121-32-no-top.h5
    include_top=False,
    input_shape=(224, 224, 3)
)
densenet.trainable = False


def save_model(_model, _model_folder, idx):
    model.save(os.path.join(_model_folder, f'model_{idx}'))
    with open(os.path.join(_model_folder, f'modelsummary_{idx}.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()


def fit_and_save(_model, _x_train, _y_train):
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max')
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    save_model(model, model_folder, i)


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
