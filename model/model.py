# pip install -U efficientnet
# pip install tensorflow --upgrade
# pip install tensorflow==2.9.1

import numpy as np 
import pandas as pd
import cv2
import numpy as np
import seaborn as sns

# pip install efficientnet
import efficientnet.keras as efn

import tensorflow as tf
from tensorflow import keras, data
from keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.applications import efficientnet, EfficientNetB0, MobileNetV2
from keras.backend import clear_session
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import random

# Setting image dimension.
IMG_HEIGHT = 224
IMG_WIDTH =  224

class DeepfakeDetectiveModel():
    def __init__(self, x_train, y_train, x_val, y_val):
        self.train_set_raw = self.create_raw_sets(x_train, y_train)
        self.val_set_raw = self.create_raw_sets(x_val, y_val)

    def create_raw_sets(self, x_set, y_set):
        return tf.data.Dataset.from_tensor_slices((x_set,y_set))

    def build_model(self):
        # Data augmentation to reduce overtraining.
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomFlip(),
                layers.RandomContrast(factor=0.1),
            ],
            name="data_augmentation",
        )

        # Initialize the base model with pre-trained ImageNet weights, and fine-tune it on the dataset.
        # Load the Base Model (using the B0 version)
        inputs = layers.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        base_model = EfficientNetB0(include_top=False, input_tensor=x, weights='imagenet')

        # Unfreeze the pretrained weights.
        base_model.trainable = True

        # Unfreeze the top 20 layers while leaving BatchNorm layers frozen.
        #for layer in base_model.layers[-20:]:
            #if not isinstance(layer, layers.BatchNormalization):
                #layer.trainable = True

        # Build the model.
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

        #model = Sequential()
        #model.add(base_model)
        #model.add(GlobalAveragePooling2D())
        #model.add(Dense(units = 1, activation = 'sigmoid'))

        # Compile.
        model = tf.keras.Model(inputs, outputs, name="EfficientNet")
        opt = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

        return model
        
    def input_preprocess(self):
        tf.keras.backend.clear_session()  # extra code – resets layer name counter

        batch_size = 32
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        train_set = self.train_set_raw.map(lambda x, y: (preprocess(tf.cast(x, tf.float32)), y))
        train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
        val_set = self.val_set_raw.map(lambda x, y: (preprocess(tf.cast(x, tf.float32)), y)).batch(batch_size)

        return(train_set, val_set)
        
    def train(self):
        # Obtain training and validation set.
        train_set, val_set = self.input_preprocess()

        model = self.build_model()
        
        earlystopping = EarlyStopping(monitor='val_accuracy', patience = 50,verbose = 1,mode = 'max')
        checkpointer = ModelCheckpoint(filepath="effnet.hdf5", verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5,min_delt=0.001,min_lr=0.00001)

        history = model.fit(
            train_set, 
            epochs=25,
            validation_data=val_set, 
            callbacks = [checkpointer, reduce_lr, earlystopping],
            verbose=1
        )
        
        self.plot_performance(history)

    def plot_performance(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.set_style("white")
        plt.suptitle('Train History', size = 15)

        ax1.plot(epochs, acc, "bo", label = "Training Accuracy")
        ax1.plot(epochs, val_acc, "b", label = "Validation Accuracy")
        ax1.set_title("Training & Validation Accuracy")
        ax1.legend()
        ax1.savefig('acc.png', bbox_inches='tight')

        ax2.plot(epochs, loss, "bo", label = "Training Loss", color = 'red')
        ax2.plot(epochs, val_loss, "b", label = "Validation Loss", color = 'red')
        ax2.set_title("Training & Validation Loss")
        ax2.legend()
        ax2.savefig('loss.png', bbox_inches='tight')

        #plt.show()