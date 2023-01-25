import pandas as pd
import cv2
from matplotlib import pyplot as plt

# 0 -> cell works
# 1 -> cell doesn't work

TEST_PATH = 'bootstrap_folds/test_folds/'
TRAIN_PATH = 'bootstrap_folds/train_folds/'


def new_file_path(current, base_folder):
    return base_folder + current.split('/')[1]


def obtain_working_cells(labels_info):
    for _, file in labels_info.iterrows():
        if file.label == 1:
            img = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
            blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
            _, clear_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(new_file_path(file.path, 'not_working_cells/'), clear_img)
            print(f'done {file.path}')


if __name__ == '__main__':
    labels_info = pd.read_csv('labels.csv', delim_whitespace=True)
    obtain_working_cells(labels_info)
