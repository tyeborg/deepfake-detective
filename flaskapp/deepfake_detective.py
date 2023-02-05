# Import the appropriate libraries.
import os
import cv2 as cv
import numpy as np

import tensorflow as tf
#import tensorflow_addons
from mtcnn.mtcnn import MTCNN
from processor import Processor
from abc import ABCMeta, abstractmethod
from moviepy.editor import VideoFileClip

class DeepfakeDetective(Processor):
    def __init__(self, input_file):
        self.input_file = input_file
        # Initialize the model for further utilization.
        self.model = tf.keras.models.load_model("./effnet", compile=False)
        # Children classes will determine the value of 'num_faces'.
        self.num_faces = None

    # Construct a method that determines if a frame is a deepfake or not.
    def analyze_individual_frame(self, frame, face):
        # Initialize face as the 'box' face parameter.
        face = face[0]['box']
        # Extract face from the frame for utilization.
        frame = super().extract_face_img(frame, face)[0]

        # Add detected rectangled face into file.
        rectangle_face = super().extract_face_img(frame, face)[1]
        #cv.imwrite('../results/detected.jpg', rectangle_face)

        # Scale and resize the face frame for EfficientNetB0 model.
        frame = super().scale_face_image(frame)
        # Utilize the model to make a prediciton upon the frame.
        result = self.model.predict(frame)[0]
        return result

    def validate(self):
        pass
        
    @abstractmethod
    def predict(self):
        pass

# Create a child class for DeepfakeDetective that handles an individual image.
class ImageInput(DeepfakeDetective):
    def __init__(self, input_file):
        super().__init__(input_file)
        # Initialize a variable that reads the image.
        self.img = super().read_img(self.input_file)

        # Locate the face and number of faces within the input frame.
        self.face = super().find_face(self.img)
        self.num_faces = super().get_num_of_faces(self.face)

    # Construct a method that performs the prediction functionality of the image.
    def predict(self):
        prediction = super().analyze_individual_frame(self.img, self.face)
        # Receive the probabilities of whether the face in the image is real or fake.
        real, deepfake = prediction[0], prediction[1]
        # Return the results
        return(real, deepfake)

# Create a child class for DeepfakeDetective that handles an individual video.
class VideoInput(DeepfakeDetective):
    def __init__(self, input_file):
        super().__init__(input_file)
        # Rename input_file to a more suitable name for this class.
        self.video = self.input_file

        # Initialize video duration and FPS.
        self.video_duration = super().get_video_length(self.video)[0]
        self.fps = super().get_video_length(self.video)[1]

        # Extract a certain amount of frames from the video (dependent on video FPS).
        self.frames = super().extract_frames(self.video, framerate=round(self.fps / 2))
        
        # Receive the number of detectable faces in the video.
        self.num_faces = super().get_num_faces_in_video(self.frames)

    def predict(self):
        # Declare lists to store thier respective probabilities.
        real, fake, unused_frames = [], [], []

        for i in range(len(self.frames)):
            # Initialize the next frame throughout each iteration.
            frame = self.frames[i]
            # Obtain the face from the frame.
            face = super().find_face(frame)

            if len(face) == 1:
                # Classify the following frame as either REAL or FAKE.
                prediction = super().analyze_individual_frame(frame, face)
                # Add the probabilities into their respective lists.
                real.append(prediction[0])
                fake.append(prediction[1])
            else:
                unused_frames.append(frame)
                continue

        # Determine the probability of the video being real and fake.
        real_probability = sum(real) / len(real)
        fake_probability = sum(fake) / len(fake)

        # Return the result.
        return real_probability, fake_probability