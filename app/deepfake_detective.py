# Run the following commands to install packages we'll use:
# pip install mtcnn
# pip install tensorflow
# pip install moviepy
# pip install python-magic-bin==0.4.14
# pip install opencv-python

# Import the appropriate libraries.
import numpy as np
import pandas as pd
import cv2 as cv
import face_recognition
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import tensorflow_addons
import os
from moviepy.editor import VideoFileClip
import magic
from abc import ABCMeta, abstractmethod

class DeepfakeDetective():
    def __init__(self, input_file):
        self.input_file = input_file
        self.detector = MTCNN()
        # Initialize the model for further utilization.
        self.model = tf.keras.models.load_model("effnet")
        # Receive file type of input file.
        self.file_type = self.get_file_type()

        # Children classes will determine the value of 'num_faces'.
        self.num_faces = None

    # Create a method that will determine if the file type of the input file.
    def get_file_type(self):
        # Identify the file type by checking input file header.
        file_type = magic.from_file(self.input_file, mime=True)
        # Split the string from the index where a slash is located.
        file_type = file_type.split("/")
        # Return the first item within the list (file type).
        return file_type[0]

    def find_face(self, frame):
        # Locate the face within the input frame.
        face = self.detector.detect_faces(frame)
        return face

    # Create a method that returns the number of faces in input image.
    def get_num_of_faces(self, face):
        return len(face)

    # Create a method that returns a given face from the image.
    def extract_and_scale_face(self, frame, face):
        # Draw a box around the discovered face.
        x, y, width, height = face[0]['box']
        x2, y2 = x + width, y + height 
        cv.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        # Structure the image to the desired format.
        face_image = frame[y:y2, x:x2]
        # Reduce the depth of the image to 1.
        face_image2 = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
        # Resize the image to size: (224, 224).
        face_image3 = cv.resize(face_image2, (224, 224))
        # Normalize the image.
        face_image4 = face_image3/255
        # Adding image to 4D.
        face_image5 = np.expand_dims(face_image4, axis=0)

        # Return the face frame viable for a Deepfake Analysis.
        return face_image5

    # Construct a method that determines if a frame is a deepfake or not.
    def analyze_individual_frame(self, frame, face):
        # Extract & properly scale the face from the image.
        frame = self.extract_and_scale_face(frame, face)

        # Utilize the model to make a prediciton upon the frame.
        result = self.model.predict(frame)[0]
        
        return result

    def validate(self):
        # Initialize a variable that will determine the validity of the image.
        error = 0
        # Initialize a message variable and a list to store those messages.
        msg, msg_lst = '', []

        # Determine if there is more than one discernable face in image.
        if self.num_faces > 1:
            error += 1
            msg = f'{self.file_type} contains too many faces'
            # Add message to the message list.
            msg_lst.append(msg)
        # Determine if there are no faces present in the image.
        if self.num_faces == 0:
            error += 1
            msg = f'{self.file_type} does not contain a discernable face'
            # Add message to the message list.
            msg_lst.append(msg)

        # Determine if any errors have applied towards the image.
        if error > 0:
            # If yes... the image cannot be submitted for deepfake prediction.
            return("error", msg_lst)
        else:
            # Else, the image has no errors & can be submitted for deepfake prediciton.
            return("success")

    @abstractmethod
    def predict(self):
        pass

# Create a child class for DeepfakeDetective that handles an individual image.
class ImageInput(DeepfakeDetective):
    def __init__(self, input_file):
        super().__init__(input_file)
        # Initialize a variable that reads the image.
        self.img = cv.imread(self.input_file)

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
        self.video_duration = self.get_video_length()[0]
        self.fps = self.get_video_length()[1]
        # Extract a certain amount of frames from the video (dependent on video FPS).
        self.frames = self.extract_frames()
        # Receive the number of detectable faces in the video.
        self.num_faces = self.get_num_faces_in_video()

    def get_video_length(self):
        clip = VideoFileClip(self.video)
        # Obtain the duration of the clip (in seconds).
        duration = clip.duration
        # Obtain the fps of the video clip.
        fps = round(clip.fps)
    
        return duration, fps

    # Create a method that extracts frames from the input video.
    def extract_frames(self):
        # Capture the input video.
        cap = cv.VideoCapture(self.video)

        frames = []
        count = 0
        
        while cap.isOpened():
            # Initialize the capture.
            ret, frame = cap.read()

            # Ensure each frame is read correctly.
            if ret:
                #cv.imwrite('./input/frames/frame{:d}.jpg'.format(count), frame)
                # Add the frame into the frames list.
                frames.append(frame)
                # Collect a frame every 1/2 frame per second.
                # For example, 24 fps --- extract every 12th frame.
                count += round(self.fps / 2) 
                cap.set(cv.CAP_PROP_POS_FRAMES, count)
            else:
                break
        
        # Esnure the video file is closed after collecting desired frames.
        cap.release()
        # Return the list that stored all the frames.
        return frames

    # Create a method that returns the number of faces within the video.
    def get_num_faces_in_video(self):
        # Initialize variable for the max number of faces.
        max_faces = 0

        # Iterate through each frame of the collection of frames.
        for frame in self.frames:
            # Determine the location of discernable faces in frame.
            face = super().find_face(frame)
            # Receive number of faces from frame.
            num_of_faces = super().get_num_of_faces(face)

            # Store the max number of faces in the max_faces variable.
            if num_of_faces > max_faces:
                max_faces = num_of_faces

        # Return the maximum number of discernable faces in the video.
        return max_faces

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