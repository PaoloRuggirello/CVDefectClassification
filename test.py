from keras.models import load_model
from common_utils import *

BASE_PATH = 'best_model'
BASE_MODEL_NAME = 'model_'
OUTPUT_FOLDER = 'best_model_metrics'


def load_keras_model(model_idx):
    print(f'Loading model: {model_idx}')
    model_name = BASE_MODEL_NAME + str(i)
    _model = load_model(os.path.join(BASE_PATH, model_name), custom_objects={'f1_m': f1_m})
    print('Model loaded')
    return _model


def load_dataset():
    print('Loading dataset')
    return np.load(os.path.join(DATA_PATH_PROCESSED, "processed_data.npy"), allow_pickle=True)


if __name__ == '__main__':
    print('Started test')
    dataset = load_dataset()
    test_folds = get_folds(TEST_PATH)

    analytics_table = dict()
    for i in range(10):
        analytics_table[str(i)] = dict()
        print(f'Iteration {i}')
        model = load_keras_model(i)
        x_test, y_test = split_samples_labels(dataset[test_folds[i].astype('uint8')])

        x_test = x_test / 255
        x_test = np.repeat(x_test[..., np.newaxis], 3, -1)

        y_pred = model.predict(x_test)
        y_pred = y_pred > 0.5

        analytics_table[str(i)]['f1'] = calculate_f1(y_test, y_pred)
        analytics_table[str(i)]['accuracy'] = calculate_accuracy(y_test, y_pred)
        print(f'F1 score: {analytics_table[str(i)]["f1"]}')
        print(f'Accuracy: {analytics_table[str(i)]["accuracy"]}')
        print("-----------------\n\n")

    sum_up_table = calculate_sum_up_table(analytics_table)
    pd.DataFrame(analytics_table).to_csv(os.path.join(OUTPUT_FOLDER, 'analytics_table.csv'))
    pd.DataFrame(sum_up_table, index=[0]).to_csv(os.path.join(OUTPUT_FOLDER, 'sum_up_table.csv'))
