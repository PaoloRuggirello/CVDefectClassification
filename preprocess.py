import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

DATA_PATH_PROCESSED = "data/processed"
DATA_IMG_W_PROCESSED = "images/w"
DATA_IMG_NOT_W_PROCESSED = "images/not_w"
DATA_IMAGES_PATH = "data/images"
SAVE_IMGS = True


def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 3)


def preprocess():
    working = 0  # label 0
    not_working = 0  # label 1
    img_size = 224

    csv_dataframe = pd.read_csv('data/labels.csv', delim_whitespace=True)
    processed_data = []
    os.makedirs(DATA_PATH_PROCESSED, exist_ok=True)

    for _, row in tqdm(csv_dataframe.iterrows()):
        image_file, label = row['path'].split('/')[1], row['label']  # reading image path and image label
        path = os.path.join(DATA_IMAGES_PATH, image_file)  # calculating image entire path
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = apply_gaussian_blur(img)
        img = cv2.resize(img, (img_size, img_size))  # resize the image

        img = img.astype('uint8')  # Reducing memory usage for the image casting variables from float64 to int8
        processed_data.append([np.array(img), label])  # Creating np.ndarray containing [image, label]

        if label:  # Counting number of working and not working images
            not_working += 1
        else:
            working += 1

    processed_data_npy = np.array(processed_data, dtype=object)
    np.save(os.path.join(DATA_PATH_PROCESSED, "processed_data.npy"), processed_data_npy)  # Saving preprocessed data inside processed_data.npy
    print("Working: ", working)
    print("Not Working: ", not_working)


if __name__ == '__main__':
    preprocess()
