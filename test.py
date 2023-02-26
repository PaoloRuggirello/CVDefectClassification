import os.path

from common_utils import *

BASE_PATH = 'best_model'
BASE_MODEL_NAME = 'model_'
OUTPUT_FOLDER = 'best_model_metrics'
WEIGHTS_FOLDER = 'best_model_weights'


def load_model(model_idx):
    print(f'Loading model {model_idx}')
    _model = get_model()
    _model.load_weights(os.path.join(WEIGHTS_FOLDER, f"model_{model_idx}.h5"))
    print('Model loaded')
    return _model


if __name__ == '__main__':
    print('Started test')
    dataset = load_dataset()
    test_folds = get_folds(TEST_PATH)  # Array containing 10-folds test info
    analytics_table = dict()

    for i in range(10):
        print(f'Iteration {i}')
        analytics_table[str(i)] = dict()
        model = load_model(i)

        x_test, y_test = split_samples_labels(dataset[test_folds[i].astype('uint32')])

        x_test = x_test / 255 # Data standardization
        x_test = np.repeat(x_test[..., np.newaxis], 3, -1)  # Creating 3-channel images from grayscale images

        y_pred = model.predict(x_test)
        y_pred = y_pred > 0.5  # Changing from Sigmoid values to binary values

        analytics_table[str(i)]['f1'] = calculate_f1(y_test, y_pred)
        analytics_table[str(i)]['accuracy'] = calculate_accuracy(y_test, y_pred)
        print(f'F1 score: {analytics_table[str(i)]["f1"]}')
        print(f'Accuracy: {analytics_table[str(i)]["accuracy"]}')
        print("-----------------\n\n")

    sum_up_table = calculate_sum_up_table(analytics_table)
    pd.DataFrame(analytics_table).to_csv(os.path.join(OUTPUT_FOLDER, 'analytics_table.csv'))
    pd.DataFrame(sum_up_table, index=[0]).to_csv(os.path.join(OUTPUT_FOLDER, 'sum_up_table.csv'))
    print('-----END----')
