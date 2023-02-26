import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage import exposure
import seam_carving


def standardize_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    return exposure.rescale_intensity(image, (0, 1), (0, 255))


DATA_PATH_PROCESSED = "data/processed"
DATA_IMG_W_PROCESSED = "images/w"
DATA_IMG_NOT_W_PROCESSED = "images/not_w"
DATA_IMAGES_PATH = "data/images"
SAVE_IMGS = True


def apply_seam_carving(_img):
    src_h, src_w, _ = _img.shape
    return seam_carving.resize(
        _img, (src_w - 76, src_h - 76),
        energy_mode='forward',  # Choose from {backward, forward}
        order='width-first',  # Choose from {width-first, height-first}
        keep_mask=None
    )


def apply_sobel(_img):
    sobel_x = cv2.Sobel(_img, cv2.CV_64F, 1, 0, ksize=5)  # x
    sobel_y = cv2.Sobel(_img, cv2.CV_64F, 0, 1, ksize=5)  # y
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    return gradient_magnitude


def apply_opening(_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(_img, cv2.MORPH_OPEN, kernel)


def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 3)


def preprocess():
    working = 0  # label 0
    not_working = 0  # label 1
    img_size = 224

    csv_dataframe = pd.read_csv('data/labels.csv', delim_whitespace=True)
    processed_data = []
    os.makedirs(DATA_PATH_PROCESSED, exist_ok=True)
    if SAVE_IMGS:
        os.makedirs(DATA_IMG_W_PROCESSED, exist_ok=True)
        os.makedirs(DATA_IMG_NOT_W_PROCESSED, exist_ok=True)

    for _, row in tqdm(csv_dataframe.iterrows()):
        image_file, label = row['path'].split('/')[1], row['label']
        path = os.path.join(DATA_IMAGES_PATH, image_file)  # concat the path
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # gaussian filter
        img = apply_gaussian_blur(img)

        img = cv2.resize(img, (img_size, img_size))  # resize the image
        img = img.astype('uint8')
        processed_data.append([np.array(img), label])

        if label:
            not_working += 1
        else:
            working += 1

        if SAVE_IMGS:
            folder = DATA_IMG_NOT_W_PROCESSED if label else DATA_IMG_W_PROCESSED
            cv2.imwrite(os.path.join(folder, image_file), img)

    processed_data_npy = np.array(processed_data, dtype=object)
    np.save(os.path.join(DATA_PATH_PROCESSED, "processed_data.npy"), processed_data_npy)
    print("Working: ", working)
    print("Not Working: ", not_working)


if __name__ == '__main__':
    preprocess()
