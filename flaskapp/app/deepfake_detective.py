# Import the appropriate libraries.
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from processor import Processor
from abc import ABCMeta, abstractmethod
from moviepy.editor import VideoFileClip
import moviepy.video.io.ImageSequenceClip

class DeepfakeDetective(Processor):
    def __init__(self, input_file):
        self.input_file = input_file
        # Initialize the model for further utilization.
        #self.model = tf.keras.models.load_model("app/effnet", compile=False)
        self.model = tf.saved_model.load("app/effnet")
        # Children classes will determine the value of 'num_faces'.
        self.num_faces = None
    
    # Construct a method that determines if a frame is a deepfake or not.
    def analyze_individual_frame(self, frame, face, input_type, count):
        # Get the input and output names.
        input_name = list(self.model.signatures['serving_default'].structured_input_signature[1].keys())[0]
        output_name = list(self.model.signatures['serving_default'].structured_outputs.keys())[0]
        
        # Initialize face as the 'box' face parameter.
        face = face[0]['box']
        # Extract face from the frame for utilization.
        frame = super().extract_face_img(frame, face, input_type, count)[0]

        # Scale and resize the face frame for EfficientNetB0 model.
        frame = super().pad_face_img(frame)
        # Utilize the model to make a prediciton upon the frame.
        result = self.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY](tf.constant(frame))[output_name]
        
        return(result)
    
    def comprehend_prediction(self, prediction):
        # Fake images = 0; Real images = 1.
        if prediction <= 1 and prediction >= 0.66:
            return('REAL')
        else:
            return('FAKE')
    
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
        # Predict the classification of the image.
        prediction = super().analyze_individual_frame(self.img, self.face, input_type='img', count=None)
        
        # Convert the prediction into meaningful information.
        results = prediction.numpy()[0][0]
        results = self.comprehend_prediction(results)
        
        return(results)

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
        
        # Initialize a counter.
        count = 0

        for i in range(len(self.frames)):
            # Initialize the next frame throughout each iteration.
            frame = self.frames[i]
            # Obtain the face from the frame.
            face = super().find_face(frame)

            if len(face) == 1:
                # Iteratively add one to the counter.
                count += 1
                
                # Classify the following frame as either REAL or FAKE.
                prediction = super().analyze_individual_frame(frame, face, input_type='video', count=count)

                # Comprehend the results in a meaningful format.
                results = prediction.numpy()[0][0]
                results = self.comprehend_prediction(results)
                
                # Based on the results, add the '1' to their respective list.
                if results == 'REAL':
                    real.append(1)
                else:
                    fake.append(1)
            else:
                unused_frames.append(frame)
                continue
        
        # Create a video out of the frames that were extracted.
        self.construct_proof_vid()
        
        # If 40% (or higher) of the video is classified as 'FAKE', then it'll
        # be classified as 'FAKE'.
        total = len(real) + len(fake)
        forty_threshold = total * 0.4
        
        if len(fake) >= forty_threshold:
            return('FAKE')
        else:
            return('REAL')
    
    def construct_proof_vid(self):
        # Initialize folder location that is storing all the frames.
        image_folder = 'app/static/files/video'
        # Set the frames per second of the video we're making.
        fps=1

        # Collect all the frames and store them within a list.
        image_files = [os.path.join(image_folder,img) for img in os.listdir(image_folder) if img.endswith(".jpg")]
        # Reverse the list contents to display in chronological order respective of the original video.
        image_files.reverse()
        
        # Make the video based off the frames at 1 frame per second.
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        # Save the video.
        clip.write_videofile('app/static/files/video/movie.mp4')