import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import threading
from skimage.exposure import rescale_intensity

# 0 -> cell works
# 1 -> cell doesn't work

TEST_PATH = 'bootstrap_folds/test_folds/'
TRAIN_PATH = 'bootstrap_folds/train_folds/'
WORKING = 0
NOT_WORKING = 1


def new_file_path(current, base_folder):
    return base_folder + current.split('/')[1]


def gaussian_otsu_threshold(img):
    blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
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
    _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    plt.imshow(img)
    return img


def standardize_image(img):
    alpha = 2.  # control contrast
    beta = -155  # control brightness

    img_new = alpha * img + beta
    return np.clip(img_new, 0, 255).astype(np.uint8)  # Given an interval, values


def obtain_working_cells(labels_info):
    for _, file in labels_info.iterrows():
        if file.label == NOT_WORKING:
            img = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
            thr_img = adaptive_threshold(img)
            image = img
            image = (image - image.min()) / (image.max() - image.min())
            bright = rescale_intensity(image, (0, 1), (0, 255))
            cv2.imwrite(new_file_path(file.path, 't4_not/'), bright)
            print(f'done {file.path}')


if __name__ == '__main__':
    labels_info = pd.read_csv('labels.csv', delim_whitespace=True)
    for label in np.array_split(labels_info, 10):
        threading.Thread(target=obtain_working_cells, args=(label,)).start()
    # obtain_working_cells(labels_info)
