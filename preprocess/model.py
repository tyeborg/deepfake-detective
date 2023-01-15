# pip install -U efficientnet
# pip install tensorflow --upgrade
# pip install tensorflow==2.9.1

import numpy as np 
import pandas as pd
import cv2
import numpy as np

# pip install efficientnet
import efficientnet.keras as efn

import tensorflow as tf
from tensorflow import keras, data
from keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D
from keras.applications import efficientnet, EfficientNetB0, MobileNetV2
from keras.backend import clear_session
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import random

# Setting image dimension.
IMG_SIZE = 224

class DeepfakeDetectiveModel():
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model = self.build()

    def build(self):
        # Data augmentation to reduce overtraining.
        data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(224, 224, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1)
        ])

        # Initialize the base model with pre-trained ImageNet weights, and fine-tune it on the dataset.
        # Load the Base Model (using the B0 version)
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

        # Freeze the model weights.
        #for layer in base_model.layers:
            #layer.trainable = False

        average_layer = GlobalAveragePooling2D()

        model = Sequential([
            data_augmentation,
            base_model,
            average_layer,
            Dense(224, activation=tf.nn.relu),
            BatchNormalization(),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid')
        ])

        earlystopping = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=25)
        checkpointer = ModelCheckpoint(filepath="effnet_test.hdf5", verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',mode='min', verbose=1, patience=10,min_delta=0.0001,factor=0.2)

        callbacks = [checkpointer, earlystopping, reduce_lr]

        # Compile and Fit.
        opt = Adam(lr=0.001)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = model.fit(
            self.x_train, self.y_train, 
            epochs = 100,
            validation_data=(self.x_val, self.y_val), 
            callbacks = callbacks,
            verbose=1)
        
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