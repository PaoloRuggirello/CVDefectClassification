import os
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

DATA_PATH = "./data"
DATA_PATH_PROCESSED = f"{DATA_PATH}/processed"
DATA_IMG_W_PROCESSED = f"{DATA_PATH_PROCESSED}/images/w"
DATA_IMG_NOT_W_PROCESSED = f"{DATA_PATH_PROCESSED}/images/not_w"
DATA_IMAGES_PATH = f"{DATA_PATH}/images"
SAVE_IMGS = True

# Preprocessing class
class ELImgPreprocessing:

    working = 0 #label 0
    not_working = 0 #label 1
    IMG_SIZE = 224

    def preprocess(self):
        csv_dataframe = pd.read_csv(os.path.join(DATA_PATH, 'labels.csv'), delim_whitespace=True)
        processed_data = []
        os.makedirs(DATA_PATH_PROCESSED, exist_ok=True)
        if SAVE_IMGS:
            os.makedirs(DATA_IMG_W_PROCESSED, exist_ok=True)
            os.makedirs(DATA_IMG_NOT_W_PROCESSED, exist_ok=True)

        for _, row in tqdm(csv_dataframe.iterrows()):
            image_file, label = row['path'].split('/')[1], row['label']
            path = os.path.join(DATA_IMAGES_PATH, image_file)  # concat the path
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # opening
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))  # resize the image
            processed_data.append([np.array(img), label]) #one-hot encoding W -> [1, 0] | NO_W -> [0, 1]

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