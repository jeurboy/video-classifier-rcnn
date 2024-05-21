# pip install git+https://github.com/tensorflow/docs

from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

dataset_path = os.listdir('dataset/train')
print(dataset_path)

room_types = os.listdir('dataset/test')
print("Types of activities found: ", len(dataset_path))

rooms = []

for item in dataset_path:
    # Get all the file names
    all_rooms = os.listdir('dataset/test' + '/' + item)

    # Add them to the list
    for room in all_rooms:
        rooms.append((item, str('dataset/test' + '/' + item) + '/' + room))

# Build a dataframe
test_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
print(test_df.head())
print(test_df.tail())

df = test_df.loc[:, ['video_name', 'tag']]
print(df)
df.to_csv('test.csv')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)
