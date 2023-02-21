import os

import pandas as pd
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from keras.applications import DenseNet121
from keras.models import Sequential
from keras import layers

from data.preprocessing import ELImgPreprocessing, DATA_PATH_PROCESSED
from contextlib import redirect_stdout
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score
from keras.callbacks import EarlyStopping


# 0 -> cell works
# 1 -> cell doesn't work

TEST_PATH = 'bootstrap_folds/test_folds/'
TRAIN_PATH = 'bootstrap_folds/train_folds/'
WORKING = 0
NOT_WORKING = 1

IN_EXAM_W = ['images/cell0004.png', 'images/cell0067.png', 'images/cell0106.png']
IN_EXAM_NOT_W = ['images/cell0165.png', 'images/cell0220.png', 'images/cell0001.png', 'images/cell0002.png', 'images/cell0057.png']

PREPROCESS = False

densenet = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
densenet.trainable = False


def new_file_path(current, base_folder):
    return base_folder + current.split('/')[1]


def gaussian_otsu_threshold(img):
    modified = img.astype("uint8")
    blur = cv2.GaussianBlur(modified, (3, 3), cv2.BORDER_DEFAULT)
    _, clear_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clear_img


def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)


def thres_finder(img, init_thres=60, delta_T=1.0):
    # Let's divide the original image into two parts
    x_low, y_low = np.where(img <= init_thres)  # Pixels values smaller than the threshold (background)
    x_high, y_high = np.where(img > init_thres)  # Pixels values greater than the threshold (foreground)

    # Find the average pixel values of the two portions
    mean_low = np.mean(img[x_low, y_low])
    mean_high = np.mean(img[x_high, y_high])

    # Calculate the new threshold by averaging the two means
    new_thres = (mean_low + mean_high) / 2

    # Stopping criteria
    if abs(new_thres - init_thres) < delta_T:  # If the difference between the previous and
        # the new threshold is less than a certain value, you have found the threshold to be applied.
        return new_thres
    else:  # Otherwise, apply the new threshold to the original image.
        return thres_finder(img, init_thres=new_thres, delta_T=5.0)


def basic_global_thresholding(img):
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return img


def standardize_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    return rescale_intensity(image, (0, 1), (0, 255))


def gradient(img):
    return cv2.Laplacian(img, cv2.CV_64F)


def obtain_working_cells(labels_info, looking_cells, allowed_images, saving_folder):
    for _, file in labels_info.iterrows():
        if file.label == looking_cells and (allowed_images is None or file.path in allowed_images):
            # img = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
            # manipulated_img = standardize_image(img)
            # manipulated_img = (gradient(manipulated_img) * 10).astype('float32')
            # manipulated_img = cv2.medianBlur(manipulated_img, 5)
            # manipulated_img = img - manipulated_img
            # manipulated_img = basic_global_thresholding(manipulated_img)
            img = cv2.imread(file.path)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            manipulated_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(new_file_path(file.path, saving_folder), manipulated_img)
            print(f'done {file.path}')


# if __name__ == '__main__':
#     labels_info = pd.read_csv('labels.csv', delim_whitespace=True)
#     for label in np.array_split(labels_info, 10):
#         threading.Thread(target=obtain_working_cells, args=(label, WORKING, None, 'intersection/')).start()
#         #threading.Thread(target=obtain_working_cells, args=(label, NOT_WORKING, None, 'not_w_grad/')).start()

def get_drop_indexes():
    result = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    row_index = np.where(np.bitwise_or.reduce(result, 1) == 0)
    column_index = np.array([0, 1, 2, 3, 4, 295, 296, 297, 298, 299])
    return row_index[0], column_index


def crop_and_save(_img, img_name, _drop_row, _drop_column):
    temp_img = []
    for r_idx, row in enumerate(_img):
        if r_idx not in _drop_row:
            temp_row = []
            for c_idx, value in enumerate(row):
                if c_idx not in _drop_column:
                    temp_row.append(value)
            temp_img.append(temp_row)
    cv2.imwrite('filtered/' + img_name, np.array(temp_img))
    print("Completed " + img_name)


def get_sorted_dict(unsortable_names):
    sortable_dict = dict()
    for name in unsortable_names:
        first_part = name.split('.')[0]
        number = first_part.split('_')[2]
        filled = number.zfill(2)
        sortable_dict[filled] = name
    return [sortable_dict[key] for key in sorted(sortable_dict)]


def get_folds(path):
    fold_files = os.listdir(path)
    fold_files = get_sorted_dict(fold_files)
    folds = []
    for file_name in fold_files:
        fold = pd.read_csv(os.path.join(path, file_name), index_col=0, delimiter=',')
        folds.append(np.array(fold[fold.columns[0]].tolist()))
        print(f'Read: {file_name}')
    return np.array(folds, dtype=object)


def save_model(_model, _model_folder, idx):
    model.save(os.path.join(_model_folder, f'model_{idx}'))
    with open(os.path.join(_model_folder, f'modelsummary_{idx}.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()


def save_sum_up_table(_model_folder, _all_f1, _all_accuracy):
    sum_up_table = dict()
    sum_up_table['f1'] = np.mean(_all_f1)
    sum_up_table['accuracy'] = np.mean(_all_accuracy)
    sum_up_table['std_accuracy'] = np.std(_all_accuracy)
    pd.DataFrame(sum_up_table, index=[0]).to_csv(os.path.join(_model_folder, 'sum_up_table.csv'))


def split_samples_labels(_dataset):
    return np.array([x for x in _dataset[:, 0]]), np.array([y for y in _dataset[:, 1]])


def get_model():
    model = Sequential()
    model.add(densenet)
    # model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.Dense(50, activation='relu'))
    # model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def fit_and_save(_model, _x_train, _y_train):
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max')
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=3,
        validation_split=0.1,
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
    save_sum_up_table(model_folder, all_f1, all_accuracy)
    print('END')

    # vari preprocessing

# drop_row, drop_column = get_drop_indexes()
# for image in os.listdir('images'):
#    img = cv2.imread('images/' + image, cv2.IMREAD_GRAYSCALE)
#    threading.Thread(target=crop_and_save, args=(img, image, drop_row, drop_column)).start()
# get row with only 0
# get columns with only 0
# these are row and columns to drop
# From real image to temp image -> take only pixels with coordinates of row - column != from column or row to drop
#   labels_info = pd.read_csv('labels.csv', delim_whitespace=True)
#   for label in np.array_split(labels_info, 10):
#       threading.Thread(target=obtain_working_cells, args=(label, WORKING, IN_EXAM_W, 'w_grad/')).start()
#       threading.Thread(target=obtain_working_cells, args=(label, NOT_WORKING, IN_EXAM_NOT_W, 'not_w_grad/')).start()
