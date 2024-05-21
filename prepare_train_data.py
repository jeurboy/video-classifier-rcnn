# pip install git+https://github.com/tensorflow/docs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dataset_path = os.listdir('dataset/train')

label_types = os.listdir('dataset/train')
print(label_types)

rooms = []

for item in dataset_path:
    # Get all the file names
    all_rooms = os.listdir('dataset/train' + '/' + item)

    # Add them to the list
    for room in all_rooms:
        rooms.append((item, str('dataset/train' + '/' + item) + '/' + room))

# Build a dataframe
train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
print(train_df.head())
print(train_df.tail())

df = train_df.loc[:, ['video_name', 'tag']]
print(df)
df.to_csv('train.csv')
