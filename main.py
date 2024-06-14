import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import numpy as np

from config import DATASET_PATH, CATEGORIES, IMAGE_SIZE, TEST_SIZE, EPOCHS, LEARNING_RATE
from utils import load_data, preprocess_data

class FaceMaskDetectionModel:
    def __init__(self):
        self.model = None

    def load_and_preprocess_data(self):
        data = load_data(DATASET_PATH, CATEGORIES, IMAGE_SIZE)
        X, y = preprocess_data(data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=TEST_SIZE)
    
    def build_model(self):
        vgg = VGG16()
        fc1_weights = vgg.get_layer('fc1').get_weights()
        fc2_weights = vgg.get_layer('fc2').get_weights()
        
        base = VGG16(include_top=False, input_shape=(*IMAGE_SIZE, 3))
        base.trainable = False
        
        self.model = Sequential()
        
        for layer in base.layers:
            self.model.add(layer)
        
        self.model.add(Flatten())
        self.model.add(Dense(25088, activation='relu', name='adjust_fc1_input'))
        self.model.add(Dense(4096, activation='relu', name='fc1'))
        self.model.layers[-1].set_weights(fc1_weights)
        self.model.add(Dense(4096, activation='relu', name='fc2'))
        self.model.layers[-1].set_weights(fc2_weights)
        self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                           loss='binary_crossentropy', metrics=['accuracy'])
    
    def train_model(self):
        self.history = self.model.fit(self.X_train, self.y_train, epochs=EPOCHS,
                                      validation_data=(self.X_test, self.y_test), verbose=1)
    
    def evaluate_model(self):
        y_prob = self.model.predict(self.X_test)
        y_pred = np.where(y_prob > 0.5, 1, 0)
        acc = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")
        print(cm)
    
    def plot_metrics(self):
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='validation_loss')
        plt.legend()
        plt.xticks(np.arange(0, EPOCHS, 1))
        plt.show()
        
        plt.plot(self.history.history['accuracy'], label='train_accuracy')
        plt.plot(self.history.history['val_accuracy'], label='validation_accuracy')
        plt.legend()
        plt.xticks(np.arange(0, EPOCHS, 1))
        plt.show()

if __name__ == '__main__':
    model = FaceMaskDetectionModel()
    model.load_and_preprocess_data()
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.plot_metrics()
    model.model.save('face_mask_detection_model.h5')
