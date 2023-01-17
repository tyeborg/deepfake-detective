import os
import math
import glob
import cv2 as cv
import numpy as np
import pandas as pd
from model import DeepfakeDetectiveModel
from sklearn.model_selection import train_test_split
from preprocess import Preprocess, FaceForensics, RealAndFakeFaces

def get_data(metadata_path):
    return pd.read_csv(metadata_path)

def get_metadatas(face_forensics_base, real_and_fake_base):
    # Create FaceForensics object.
    face_forensics = FaceForensics(face_forensics_base)
    # Extract frames and allocate them to their respective directories.
    face_forensics.execute_extract_allocation()
    face_forensics_df = face_forensics.meta
    # Convert dataframe into a csv file.
    face_forensics_df.to_csv(face_forensics_base + 'metadata.csv', index=False)

    # Create RealAndFakeFaces object.
    real_and_fake = RealAndFakeFaces(real_and_fake_base)
    real_and_fake_df = real_and_fake.meta
    # Convert dataframe into a csv file.
    real_and_fake_df.to_csv(real_and_fake_base + 'metadata.csv', index=False)

def main():
    # Declare the basepaths for our datasets.
    face_forensics_base = './datasets/face_forensics/'
    real_and_fake_base = './datasets/real_and_fake_face_detection/'
    
    # Use this code snippet to create metadata from the datasets.
    #get_metadatas(face_forensics_base, real_and_fake_base)

    # Create a Preprocess object.
    preprocess = Preprocess()

    # Read the Face Forensics metadata.
    face_forensics_df = get_data(face_forensics_base + 'metadata.csv')
    # Read the Real and Fake Faces Detection metadata.
    real_and_fake_df = get_data(real_and_fake_base + 'metadata.csv')

    # Receive a merged and sampled meta.
    sampled_meta = preprocess.get_sampled_meta(face_forensics_df, real_and_fake_df)

    # Export result in csv.
    sampled_meta.to_csv('./datasets/final_metadata.csv', index=False)

    # Obtain training, testing, and validation sets from 'data_fused_meta'.
    train_set, test_set = train_test_split(sampled_meta,test_size=0.2,random_state=42,stratify=sampled_meta['label'])
    train_set, val_set = train_test_split(train_set,test_size=0.3,random_state=42,stratify=train_set['label'])

    # Segmentize sets between the images (x) and labels (y).
    x_train, y_train = preprocess.get_segment_sets(train_set)
    x_val, y_val = preprocess.get_segment_sets(val_set)
    x_test, y_test = preprocess.get_segment_sets(test_set)

    effnet = DeepfakeDetectiveModel(x_train, y_train, x_val, y_val)
    effnet.train()

if __name__ =='__main__':
    main()