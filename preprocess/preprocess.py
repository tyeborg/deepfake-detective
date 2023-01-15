import ast
import random
import string
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from mtcnn.mtcnn import MTCNN

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath('../app/processor.py'))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from app.processor import Processor

class Preprocess(Processor):
    def __init__(self):
        self.basepath = None
        self.deepfake_dir = None
        self.real_dir = None

    def split_meta(self, meta):
        # Separate real images from the fake images 
        # into their own separate dataframes.
        real_df = meta[meta["label"] == "REAL"]
        fake_df = meta[meta["label"] == "FAKE"]

        return(real_df, fake_df)

    def get_min_df_size(self, real_df, fake_df):
        # Initialize a variable that will set the minimum length amount.
        min_len = 0

        # Determine the length of each dataframe.
        real_len, fake_len = len(real_df), len(fake_df)

        # Determine which length has the lesser value
        # and store it inside the min_len variable.
        if real_len < fake_len:
            min_len = real_len
        else:
            min_len = fake_len

        return min_len

    def get_sampled_meta(self, face_forensics_df, real_and_fake_df):
        # Split dataframes between real and fake labels.
        real_forensics, fake_forensics = self.split_meta(face_forensics_df)
        real_r_and_f, fake_r_and_f = self.split_meta(real_and_fake_df)

        # Obtain the minimum sizes between real and fake df's of original df's. 
        face_forensics_sample_size = self.get_min_df_size(real_forensics, fake_forensics)
        real_and_fake_sample_size = self.get_min_df_size(real_r_and_f, fake_r_and_f)

        # Receive an appropriate sample size (balanced between two datasets).
        sample_size = 0
        if face_forensics_sample_size < real_and_fake_sample_size:
            sample_size = face_forensics_sample_size
        else:
            sample_size = real_and_fake_sample_size

        # Set split dataframes to the size set within 'sample_size'.
        real_forensics = real_forensics.sample(sample_size, random_state=42)
        fake_forensics = fake_forensics.sample(sample_size, random_state=42)
        real_r_and_f = real_r_and_f.sample(sample_size, random_state=42)
        fake_r_and_f = fake_r_and_f.sample(sample_size, random_state=42)

        # Respectively merge real and fake dataframes.
        face_forensics_df = pd.concat([real_forensics, fake_forensics])
        real_and_fake_df = pd.concat([real_r_and_f, fake_r_and_f])

        # Merge the two sampled datasets together
        data_merged_meta = pd.concat([face_forensics_df, real_and_fake_df])
        return data_merged_meta

    def get_segment_sets(self, set_name):
        # Initialize lists to store image data and label data.
        images, labels = [], []

        for (img_path, face, label) in zip(set_name['path'], set_name["face"], set_name['label']):
            # Read the image.
            img = super().read_img(img_path)

            # Converting face string to list.
            face = ast.literal_eval(face)

            # Add the formatted image into the images array.
            images.append(super().format_img(img, face))

            # Fake images = 0; Real images = 1.
            if (label == 'FAKE'):
                labels.append(0)
            else:
                labels.append(1)

        # Return a the image data along with each images' label.
        return np.array(images, dtype='float')/255.0, np.array(labels)

    def get_label(self, path):
        if path == self.deepfake_dir:
            label = "FAKE"
        else:
            label = "REAL"

        return label

    def real_fake_fuse(self):
        # Obtain the dataframe full of deepfakes.
        deepfake_df = self.make_meta(self.deepfake_dir)
        # Obtain the dataframe full of real faces.
        real_df = self.make_meta(self.real_dir)

        # Combine deepfake_df and real_df
        merged_df = deepfake_df.append(real_df, ignore_index=True)

        return merged_df

    def make_meta(self, path):
        # Initialize a dictionary to store info.
        meta = {"path": [], "face": [], "label": []}

        label = self.get_label(path)

        # Iterate through every file within the directory/path.
        for img in os.listdir(path):
            # Ensure the files have a JPG extension.
            if img.endswith(".jpg"):
                # Set path for the image.
                img_path = path + img

                # Read the image.
                img = super().read_img(img_path)

                # Use MTCNN face detector to extract face from img.
                face = super().find_face(img)
                # Get the number of faces detected in the image.
                num_of_faces = super().get_num_of_faces(face)
            
                # Append to the meta dictionary if only one face was detected.
                if num_of_faces == 1:
                    meta['path'].append(img_path)
                    meta["face"].append(face[0]['box'])
                    meta['label'].append(label)
                else:
                    continue
            else:
                continue

        # Convert dictionary to a dataframe and return it.
        meta = pd.DataFrame(meta)
        return meta

class FaceForensics(Preprocess):
    def __init__(self, basepath):
        self.basepath = basepath
        self.deepfake_dir = self.basepath + 'img_data/deepfake/'
        self.real_dir = self.basepath + 'img_data/real/'

        self.meta = None

    def execute_extract_allocation(self):
        # Initialize paths of the directories that contain the videos (for frame extraction).
        video_deepfake_dir = self.basepath + 'video_data/deepfakes/'
        video_real_dir = self.basepath + 'video_data/original_sequences/'

        # Extract frames and allocate from deepfake video data.
        self.extract_and_allocate_frames(video_deepfake_dir)
        # Extract frames and allocate from the real video data.
        self.extract_and_allocate_frames(video_real_dir)

        self.meta = super().real_fake_fuse()

    # Extract frames from videofile path.
    def extract_and_allocate_frames(self, path):
        # Determine the label based on the path.
        if path == self.basepath + 'video_data/deepfakes/':
            label = "FAKE"
        else:
            label = "REAL"

        # Iterate through every file within the directory/path.
        for video in os.listdir(path):
            # Ensure the files have an MP4 or MOV extension.
            if video.endswith(".mp4") or video.endswith(".mov"):
                # Declare the proper path to the video.
                video = path + video
                # Obtain the video FPS.
                fps = super().get_video_length(video)[1]
                # Initilaize the a variable that determines that extraction time of each frame.
                framerate = fps * 4

                # Obtain every frame per second within the video.
                frames = super().extract_frames(video, framerate)

                # Store the frames respectively.
                self.store_frames(frames, label)
            
    def generate_id(self, size=20, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))
        
    def store_frames(self, frames, label):
        # Initialize a variable for the frame path.
        storage_path = self.basepath + 'img_data/'

        if label == "FAKE":
            storage_path = storage_path + 'deepfake/'
        else:
            storage_path = storage_path + 'real/'

        for frame in frames:
            # Generate a random filename for the frame.
            frame_name = self.generate_id() + '.jpg'
            # Set the frame path.
            frame_path = storage_path + frame_name

            # Save frame in suggested 'frame_path'.
            cv.imwrite(frame_path, frame)

class RealAndFakeFaces(Preprocess):
    def __init__(self, basepath):
        self.basepath = basepath
        self.deepfake_dir = self.basepath + 'img_data/deepfake/'
        self.real_dir = self.basepath + 'img_data/real/'
        self.meta = self.real_fake_fuse()