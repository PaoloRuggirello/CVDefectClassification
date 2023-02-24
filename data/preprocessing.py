import os
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage import exposure
import seam_carving
from skimage.feature import local_binary_pattern
from PIL import Image


def standardize_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    return exposure.rescale_intensity(image, (0, 1), (0, 255))


DATA_PATH_PROCESSED = "processed"
DATA_IMG_W_PROCESSED = "images/w"
DATA_IMG_NOT_W_PROCESSED = "images/not_w"
DATA_IMAGES_PATH = "images"
SAVE_IMGS = True


# Preprocessing class
class ELImgPreprocessing:
    working = 0  # label 0
    not_working = 0  # label 1
    IMG_SIZE = 224

    def apply_seam_carving(self, _img):
        src_h, src_w = _img.shape
        return seam_carving.resize(
            _img, (src_w - 76, src_h - 76),
            energy_mode='forward',  # Choose from {backward, forward}
            order='width-first',  # Choose from {width-first, height-first}
            keep_mask=None
        )

    def apply_sobel(self, _img):
        sobel_x = cv2.Sobel(_img, cv2.CV_64F, 1, 0, ksize=5)  # x
        sobel_y = cv2.Sobel(_img, cv2.CV_64F, 0, 1, ksize=5)  # y
        gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        gradient_magnitude *= 255.0 / gradient_magnitude.max()
        return gradient_magnitude

    def apply_opening(self, _img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        return cv2.morphologyEx(_img, cv2.MORPH_OPEN, kernel)

    def preprocess(self):
        csv_dataframe = pd.read_csv('labels.csv', delim_whitespace=True)
        processed_data = []
        processed_data_lbp = []
        os.makedirs(DATA_PATH_PROCESSED, exist_ok=True)
        if SAVE_IMGS:
            os.makedirs(DATA_IMG_W_PROCESSED, exist_ok=True)
            os.makedirs(DATA_IMG_NOT_W_PROCESSED, exist_ok=True)

        for _, row in tqdm(csv_dataframe.iterrows()):
            image_file, label = row['path'].split('/')[1], row['label']
            path = os.path.join(DATA_IMAGES_PATH, image_file)  # concat the path
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # seam carving
            img = self.apply_seam_carving(img)

            #sobel
            img = self.apply_sobel(img)

            #sum + sobel + image
            # img = img + sobel
            # img[np.where(img > 255)] = 255

            #opening
            #img = self.apply_opening(img)

            # # opening
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # # end opening

            # standardize
            # img = standardize_image(img)



            # img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))  # resize the image
            # lbp_img = local_binary_pattern(img, 8, 1).astype('uint8')
            # sobel = self.apply_sobel(img).astype('float16')

            # processed_data.append([np.array([img, lbp_img, sobel]).transpose(1, 2, 0), label])
            img = img.astype('uint8')
            processed_data.append([np.array(img), label])

            if label:
                self.not_working += 1
            else:
                self.working += 1

            if SAVE_IMGS:
                folder = DATA_IMG_NOT_W_PROCESSED if label else DATA_IMG_W_PROCESSED
                cv2.imwrite(os.path.join(folder, image_file), img)

        processed_data_npy = np.array(processed_data, dtype=object)
        np.save(os.path.join(DATA_PATH_PROCESSED, "processed_data.npy"), processed_data_npy)
        print("Working: ", self.working)
        print("Not Working: ", self.not_working)


if __name__ == '__main__':
    preprocessing = ELImgPreprocessing()
    preprocessing.preprocess()
