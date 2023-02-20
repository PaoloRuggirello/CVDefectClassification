import os

import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import threading
from skimage.exposure import rescale_intensity

from data.preprocessing import ELImgPreprocessing, DATA_PATH_PROCESSED

# 0 -> cell works
# 1 -> cell doesn't work

TEST_PATH = 'bootstrap_folds/test_folds/'
TRAIN_PATH = 'bootstrap_folds/train_folds/'
WORKING = 0
NOT_WORKING = 1

IN_EXAM_W = ['images/cell0004.png', 'images/cell0067.png', 'images/cell0106.png']
IN_EXAM_NOT_W = ['images/cell0165.png', 'images/cell0220.png', 'images/cell0001.png', 'images/cell0002.png', 'images/cell0057.png']


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


def oneHotErrorRaiser(oneHotEncoding):
    raise ValueError(f'Invalid one hot encoding: {oneHotEncoding}')


def oneHot2Label(oneHotEncoding):
    # 0 -> WORKING
    # 1 -> NOT WORKING
    return 0 if (oneHotEncoding == np.eye(2)[0]).all() else 1 if (oneHotEncoding == np.eye(2)[1]).all() else oneHotErrorRaiser(oneHotEncoding=oneHotEncoding)


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


if __name__ == '__main__':
    preprocessing = ELImgPreprocessing()
    preprocessing.preprocess()
    dataset = np.load(os.path.join(DATA_PATH_PROCESSED, "processed_data.npy"), allow_pickle=True)
    train_folds = get_folds(TRAIN_PATH)
    test_folds = get_folds(TEST_PATH)

    for i in range(10):
        print(f'Iteration: {i}')
        train_fold = dataset[train_folds[i].astype(np.int64)]
        test_fold = dataset[test_folds[i].astype(np.int64)]

        print(train_fold.size)
        print(test_fold.size)
        print("-----------------\n\n")

    print('END')
    # test_folds = get_folds(TEST_PATH)
# TODO -> rename train and test folds


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
