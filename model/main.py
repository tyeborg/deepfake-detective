import os
import math
import glob
#import cv2 as cv
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot
from model import DeepfakeDetectiveModel
from preprocess import Preprocess, FaceForensics
from sklearn.model_selection import train_test_split

def plot_sets(Train_set, Val_set, Test_set):
    y = dict()

    y[0] = []
    y[1] = []

    for set_name in (np.array(Train_set['label']), np.array(Val_set['label']), np.array(Test_set['label'])):
        y[0].append(np.sum(set_name == 'REAL'))
        y[1].append(np.sum(set_name == 'FAKE'))

    trace0 = go.Bar(
        x=['Train Set', 'Validation Set', 'Test Set'],
        y=y[0],
        name='REAL',
        marker=dict(color='#2696C2'),
        opacity=0.7
    )
    trace1 = go.Bar(
        x=['Train Set', 'Validation Set', 'Test Set'],
        y=y[1],
        name='FAKE',
        marker=dict(color='#A5104B'),
        opacity=0.7
    )

    data = [trace0, trace1]
    layout = go.Layout(
        title='Count of Images in Each Set',
        xaxis={'title': 'Set'},
        yaxis={'title': 'Count'}
    )

    fig = go.Figure(data, layout)
    iplot(fig)

def get_data(metadata_path):
    return pd.read_csv(metadata_path)

def get_metadatas(face_forensics_base):
    # Create FaceForensics object.
    face_forensics = FaceForensics(face_forensics_base)
    # Extract frames and allocate them to their respective directories.
    face_forensics.execute_extract_allocation()
    face_forensics_df = face_forensics.meta
    # Convert dataframe into a csv file.
    face_forensics_df.to_csv(face_forensics_base + 'metadata.csv', index=False)

def main():
    # Declare the basepaths for our datasets.
    face_forensics_base = './datasets/face_forensics/'
    
    # Use this code snippet to create metadata from the datasets.
    #get_metadatas(face_forensics_base, real_and_fake_base)

    # Create a Preprocess object.
    preprocess = Preprocess()

    # Read the Face Forensics metadata.
    #face_forensics_df = get_data(face_forensics_base + 'metadata.csv')

    # Export result in csv.
    #sampled_meta.to_csv('./datasets/final_metadata.csv', index=False)
    sampled_meta = pd.read_csv(face_forensics_base + 'metadata.csv')

    real_df = sampled_meta[sampled_meta["label"] == "REAL"]
    fake_df = sampled_meta[sampled_meta["label"] == "FAKE"]
    sample_size = 8500

    real_df = real_df.sample(sample_size, random_state=42)
    fake_df = fake_df.sample(sample_size, random_state=42)

    sampled_meta = pd.concat([real_df, fake_df])

    # Obtain training, testing, and validation sets from 'data_fused_meta'.
    train_set, test_set = train_test_split(sampled_meta,test_size=0.30,random_state=42,stratify=sampled_meta['label'])
    test_set, val_set = train_test_split(test_set,test_size=0.5,random_state=42,stratify=test_set['label'])

    print(f'Train set: {train_set.shape}')
    print(f'Validation set: {val_set.shape}')
    print(f'Test set: {test_set.shape}')

    plot_sets(train_set, val_set, test_set)

    # Segmentize sets between the images (x) and labels (y).
    x_train, y_train = preprocess.get_segment_sets(train_set, name="Train Set")
    x_val, y_val = preprocess.get_segment_sets(val_set, name="Validation Set")
    x_test, y_test = preprocess.get_segment_sets(test_set, name="Test Set")

    effnet = DeepfakeDetectiveModel(x_train, y_train, x_val, y_val, x_test, y_test)
    effnet.train()

    #custom_cnn = CustomCNNModel(x_train, y_train, x_val, y_val, x_test, y_test)
    #custom_cnn.train()

if __name__ =='__main__':
    main()