import os 
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D
from tensorflow.keras import preprocessing
from sklearn.utils import shuffle  
import pandas as pd
import matplotlib.pyplot as plt

# pip install -U efficientnet
# pip install tensorflow --upgrade

class DeepfakeDetectiveModel():
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.model = self.build()

    def build(self):
        # Data augmentation to reduce overtraining.
        data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(224, 224, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1)
        ])

        model = tf.keras.Sequential([
            data_augmentation,
            tf.keras.applications.EfficientNetB0(
                input_shape=(224,224,3),
                weights='imagenet',
                include_top=False,
                drop_connect_rate=0.5
            ),
            GlobalAveragePooling2D(),
            Dense(units=1, activation='sigmoid')
        ])

        model.compile(
            optimizer = 'adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy']
            )

        history = model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs= 20
        )

        self.plot_performance(history)

        return model

    def plot_performance(self, history):
        # Plotting Results
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.title('Training & Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        #fig = plt.figure()
        plt.savefig('acc.png', bbox_inches='tight')

        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'g', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')

        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig('loss.png', bbox_inches='tight')