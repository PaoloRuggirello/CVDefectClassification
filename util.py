import numpy as np
import seam_carving
import cv2
import os
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage.feature import local_binary_pattern
import Augmentor
from keras import backend as K


def new_file_path(current, base_folder):
    return base_folder + current.split('/')[1]


def gaussian_otsu_threshold(img):
    modified = img.astype("uint8")
    blur = cv2.GaussianBlur(modified, (3, 3), cv2.BORDER_DEFAULT)
    _, clear_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clear_img


def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)


#
# def thres_finder(img, init_thres=60, delta_T=1.0):
#     # Let's divide the original image into two parts
#     x_low, y_low = np.where(img <= init_thres)  # Pixels values smaller than the threshold (background)
#     x_high, y_high = np.where(img > init_thres)  # Pixels values greater than the threshold (foreground)
#
#     # Find the average pixel values of the two portions
#     mean_low = np.mean(img[x_low, y_low])
#     mean_high = np.mean(img[x_high, y_high])
#
#     # Calculate the new threshold by averaging the two means
#     new_thres = (mean_low + mean_high) / 2
#
#     # Stopping criteria
#     if abs(new_thres - init_thres) < delta_T:  # If the difference between the previous and
#         # the new threshold is less than a certain value, you have found the threshold to be applied.
#         return new_thres
#     else:  # Otherwise, apply the new threshold to the original image.
#         return thres_finder(img, init_thres=new_thres, delta_T=5.0)


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

# def get_drop_indexes():
#     result = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
#     row_index = np.where(np.bitwise_or.reduce(result, 1) == 0)
#     column_index = np.array([0, 1, 2, 3, 4, 295, 296, 297, 298, 299])
#     return row_index[0], column_index


# def crop_and_save(_img, img_name, _drop_row, _drop_column):
#     temp_img = []
#     for r_idx, row in enumerate(_img):
#         if r_idx not in _drop_row:
#             temp_row = []
#             for c_idx, value in enumerate(row):
#                 if c_idx not in _drop_column:
#                     temp_row.append(value)
#             temp_img.append(temp_row)
#     cv2.imwrite('filtered/' + img_name, np.array(temp_img))
#     print("Completed " + img_name)

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

def apply_seam_carving(_img):
    src_h, src_w = _img.shape
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


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == '__main__':
    # for image in os.listdir('data/images'):

    f1_m(1, 0.6)
    # image = 'cell0001.png'
    # image1 = 'cell1922.png'
    # image2 = 'cell0266.png'
    # image3 = 'cell0079.png'
    #
    # print(f'File: {image}')
    # img = cv2.imread('data/images/' + image, cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.imread('data/images/' + image1, cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('data/images/' + image2, cv2.IMREAD_GRAYSCALE)
    # img3 = cv2.imread('data/images/' + image3, cv2.IMREAD_GRAYSCALE)
    # imgs = np.array([img, img1, img2, img3])
    #
    # for i in range(len(imgs)):
    #     img = standardize_image(imgs[i])
    #     input("Press enter for next image")
    #     Image.fromarray(img).show()
    #     sobel = apply_sobel(img)
    #     opened = apply_opening(img + sobel)
    #     opened[np.where(opened > 255)] = 255
    #     # Image.fromarray(sobel).show()
    #     # Image.fromarray(img - sobel).show(title='Sub')
    #     Image.fromarray(opened).show(title='Add')
    #     Image.fromarray(apply_sobel(opened)).show()
    #
    #     # break
