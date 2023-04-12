import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
from moviepy.editor import VideoFileClip
from tensorflow.keras.preprocessing.image import img_to_array

# Setting image dimension.
IMG_WIDTH = 224
IMG_HEIGHT = 224

class Processor():
    def __init__(self):
        pass
        
    def read_img(self, img_path):
        # Read the image using cv2 (renamed to cv).
        img = cv.imread(img_path)
        return img

    # Create a method that returns a given face from the image.
    def extract_face_img(self, img, face, input_type, count):
        # Draw a box around the discovered face.
        x, y, width, height = face
        x2, y2 = x + width, y + height 

        cv.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        
        # Store the boxed face images in the respective folders (img or video).
        if input_type == 'img':
            cv.imwrite('app/static/files/img/frame.jpg', img)
        elif input_type == 'video':
            cv.imwrite('app/static/files/video/frame{:d}.jpg'.format(count), img)
        else:
            pass

        # Structure the image to the desired format.
        face_img = img[y:y2, x:x2]
        #cv.imwrite('app/static/detected.jpg', img)
        return(face_img, img)

    # Create a method that rescales the image of the extracted face.
    def scale_face_image(self, face_img):
        # Convert it to the right color format.
        face_img = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
        
        # Resize the image based on the input dimensions: (224, 224).
        face_img = cv.resize(face_img, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv.INTER_AREA)

        # Convert the image to a numpy array with float32 as the datatype.
        face_img = np.array(face_img)
        face_img = face_img.astype('float32')

        # Normalize the image.
        face_img /= 255

        # Adding image to 4D.
        face_img = np.expand_dims(face_img, 0)

        # Return the frame for a Deepfake Analysis.
        return face_img

    def format_img(self, img, face):
        # Extract the face within the given image.
        img = self.extract_face_img(img, face)
        # Scale and resize the image to fit the format for EfficientNetB0 model.
        img = self.scale_face_image(img)
        return img

    def find_face(self, frame):
        # Use a consistent face recongition detector across all datasets.
        detector = MTCNN()
        # Locate the face within the input frame.
        face = detector.detect_faces(frame)
        return face

    # Create a method that returns the number of faces in input image.
    def get_num_of_faces(self, face):
        return len(face)

    # Create a method that returns the number of faces within the video.
    def get_num_faces_in_video(self, frames):
        # Initialize variable for the max number of faces.
        max_faces = 0

        # Iterate through each frame of the collection of frames.
        for frame in frames:
            # Determine the location of discernable faces in frame.
            face = self.find_face(frame)
            # Receive number of faces from frame.
            num_of_faces = self.get_num_of_faces(face)

            # Store the max number of faces in the max_faces variable.
            if num_of_faces > max_faces:
                max_faces = num_of_faces

        # Return the maximum number of discernable faces in the video.
        return max_faces

    def get_video_length(self, video):
        clip = VideoFileClip(video)
        # Obtain the duration of the clip (in seconds).
        duration = clip.duration
        # Obtain the fps of the video clip.
        fps = round(clip.fps)
    
        return(duration, fps)

    # Create a method that extracts frames from the input video.
    def extract_frames(self, video, framerate):
        # Capture the input video.
        cap = cv.VideoCapture(video)

        frames, count = [], 0
        
        while cap.isOpened():
            # Initialize the capture.
            ret, frame = cap.read()

            # Ensure each frame is read correctly.
            if ret:
                # Add the frame into the frames list.
                frames.append(frame)
                # Collect a frame every 1/2 frame per second.
                # For example, 24 fps --- extract every 12th frame.
                count += framerate
                cap.set(cv.CAP_PROP_POS_FRAMES, count)
            else:
                break
        
        # Esnure the video file is closed after collecting desired frames.
        cap.release()
        # Return the list that stored all the frames.
        return frames