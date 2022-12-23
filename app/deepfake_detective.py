# Run the following commands to install packages we'll use:
# pip install mtcnn
# pip install tensorflow
# pip install moviepy

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

# Create a class that collects all frames from input video.
class DeepfakeDetective():
    def __init__(self, input_video):
        self.video_file = input_video
        self.video_duration = self.get_video_duration()[0]
        self.fps = self.get_video_duration()[1]

        self.frames = self.extract_frames()
        self.detector = MTCNN()

        # Validation variables:
        self.num_faces_in_video = self.get_num_faces_in_video()

    # Create a method that extracts frames from the input video.
    def extract_frames(self):
        # Capture the input video.
        cap = cv.VideoCapture(self.video_file)

        frames = []
        count = 0

        frames_dir = './frames'
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
        
        while cap.isOpened():
            # Initialize the capture.
            ret, frame = cap.read()

            # Ensure each frame is read correctly.
            if ret:
                cv.imwrite('./frames/frame{:d}.jpg'.format(count), frame)
                # Add the frame into the frames list.
                frames.append(frame)
                # Collect a frame every 0.5 frame per second.
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
            face = self.detector.detect_faces(frame)
            num_of_faces = len(face)

            # Store the max number of faces in the max_faces variable.
            if num_of_faces > max_faces:
                max_faces = num_of_faces
        
        # Return the maximum number of discernable faces in the video.
        return max_faces

    def get_video_duration(self):
        clip = VideoFileClip(self.video_file)
        duration = clip.duration
        fps = round(clip.fps)
    
        return duration, fps

    # We need to ensure the video contains one person and is less than 2mins.
    def validate_video(self):
        # Initialize variables that will determine the validity of the video.
        success = 0
        error = 0

        # Determine if there is more than one discernable face in video.
        if self.num_faces_in_video > 1:
            error += 1

        # Determine if the video surpasses the 2 minute mark (120 seconds).
        if self.video_duration > 120:
            error += 1

        # Determine if any errors have applied towards the video.
        if error > 0:
            # If yes... the video cannot be submitted for deepfake prediction.
            success = 0
        else:
            # Else, the video has no errors & can be submitted for deepfake prediciton.
            success += 1

        # Return something...