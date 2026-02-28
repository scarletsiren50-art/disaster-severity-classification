import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_processed_dataset(base_path="data/processed"):

    X = []
    y = []
    class_names = os.listdir(base_path)

    class_names.sort()  # important for label consistency

    for label, class_name in enumerate(class_names):

        class_path = os.path.join(base_path, class_name)

        for img_name in os.listdir(class_path):

            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y), class_names