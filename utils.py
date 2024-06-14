import os
import cv2
import numpy as np

def load_data(dataset_path, categories, image_size):
    data = []
    for category in categories:
        path = os.path.join(dataset_path, category)
        label = categories.index(category)
        
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            data.append([img, label])
    
    return data

def preprocess_data(data):
    X = []
    y = []
    for feature, label in data:
        X.append(feature)
        y.append(label)
    X = np.array(X) / 255.0
    y = np.array(y)
    return X, y
