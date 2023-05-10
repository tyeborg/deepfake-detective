import numpy as np 
import pandas as pd
import seaborn as sns
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from tensorflow.data import Dataset
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications import efficientnet, EfficientNetB0
from tensorflow.keras.layers import Input, RandomRotation, RandomTranslation, RandomFlip, RandomContrast
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten

# Setting image dimension.
IMG_HEIGHT = 224
IMG_WIDTH =  224

class MyModel():
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def display_confusion_matrix(self, cm):
        sns.set(font_scale=1.2)
        sns.set_style("white")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=True)
        plt.xlabel("Predicted Labels", fontsize=14)
        plt.ylabel("True Labels", fontsize=14)
        plt.title("Model Applied Towards Test Set", fontsize=16)
        plt.show()

    def plot_performance(self, model_fit, loss_ylim_min, loss_ylim_max):
        acc = model_fit.history['accuracy']
        val_acc = model_fit.history['val_accuracy']
        loss_ = model_fit.history['loss']
        val_loss_ = model_fit.history['val_loss']

        # Get the number of epochs from the history object
        num_epochs = len(model_fit.history['loss'])

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.xlim([1, num_epochs])
        plt.ylim([0.5, 1])

        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss_, label='Training Loss')
        plt.plot(val_loss_, label='Validation Loss')
        plt.xlim([1, num_epochs])
        plt.ylim([loss_ylim_min, loss_ylim_max])
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.savefig('epoch-performance.png', bbox_inches='tight')

class DeepfakeDetectiveModel(MyModel):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        super().__init__(x_train, y_train, x_val, y_val, x_test, y_test)

    def build_model(self):
        # Initialize the base model with pre-trained ImageNet weights, and fine-tune it on the dataset.
        # Load the Pretrained Base Model (using the B0 version).
        tf.random.set_seed(42)
        base_model = EfficientNetB0(weights="imagenet", include_top=False)

        # Unfreeze the last 75 pre-trained layers.
        for layer in base_model.layers[-75:]:
            layer.trainable = True

        # Add new layers for classification
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        # Compile.
        model = Model(inputs=base_model.input, outputs=outputs)
        opt = Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

        return model
        
    def train(self):
        # Receive the compiled model.
        model = self.build_model()

        # Define early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
        
        # Fit the model.
        history = model.fit(
            self.x_train, 
            self.y_train,
            validation_data=(self.x_val, self.y_val), 
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        # Save the model.
        #model.save("effnet.h5")
        tf.saved_model.save(model, 'effnet')

        # Evaluate the model on the test data.
        test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test)

        # Print the test loss and accuracy.
        print('Test loss:', test_loss)
        print('Test accuracy:', test_accuracy)

        # Plot and save the model performance.
        super().plot_performance(history, loss_ylim_min=0, loss_ylim_max=0.8)

        y_pred = model.predict(self.x_test)
        # Convert to binary using threshold of 0.5.
        y_pred_binary = (y_pred > 0.5).astype(int)

        cm = confusion_matrix(self.y_test, y_pred_binary)
        super().display_confusion_matrix(cm)

class CustomCNNModel(MyModel):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        super().__init__(x_train, y_train, x_val, y_val, x_test, y_test)

    def build_model(self):
        tf.random.set_seed(42)

        # Define the input shape of the image
        input_shape = (224, 224, 3)

        # Define the Sequential model
        model = tf.keras.Sequential([
            Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # Compile.
        opt = Adam()
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        return model

    def train(self):
        # Receive the compiled model.
        model = self.build_model()

        # Define early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
        
        # Fit the model.
        history = model.fit(
            self.x_train, 
            self.y_train,
            validation_data=(self.x_val, self.y_val), 
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate the model on the test data.
        test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test)

        # Print the test loss and accuracy.
        print('Test loss:', test_loss)
        print('Test accuracy:', test_accuracy)

        # Plot and save the model performance.
        super().plot_performance(history, loss_ylim_min=0, loss_ylim_max=0.8)

        y_pred = model.predict(self.x_test)
        # Convert to binary using threshold of 0.5.
        y_pred_binary = (y_pred > 0.5).astype(int)

        cm = confusion_matrix(self.y_test, y_pred_binary)
        super().display_confusion_matrix(cm)