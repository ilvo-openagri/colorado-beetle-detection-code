import os
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import constants
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
import cv2
import seaborn as sn
import pandas as pd
import matplotlib.patches as patches
import time

def img_to_512(image):
    # height = len(image)
    # width = len(image[0])
    # size_piece = 512
    labels = []
    pieces = []
    y = 0
    while(y < constants.Y_END - constants.WIDTH_PIECE): #zijkanten worden weggegooid indien te klein
        x = 0
        while(x < constants.X_END - constants.WIDTH_PIECE): #zijkanten worden weggegooid indien te klein
            x_end = x+constants.WIDTH_PIECE if x+constants.WIDTH_PIECE < constants.X_END else constants.X_END
            y_end = y+constants.WIDTH_PIECE if y+constants.WIDTH_PIECE < constants.Y_END else constants.Y_END
            #cut a piece
            img_piece = image[y:y_end,x:x_end]
            pieces.append(img_piece)
            coordinate = [x + constants.WIDTH_PIECE / 2, y + constants.WIDTH_PIECE / 2]
            labels.append(coordinate)
            x += constants.WIDTH_PIECE
        y += constants.WIDTH_PIECE
    
    return labels, pieces

def filter_512(data, labels, model):
    filtered_data = []
    filtered_label = []
    aantal = len(data)

    for i in range(aantal):
        img = skimage.transform.resize(data[i], (64,64))
        img = img.reshape((1,-1))
        # print(model.predict(img))
        prediction = model.predict(img)
        if prediction == 1:
            filtered_data.append(data[i])
            filtered_label.append(labels[i])
        print(f'    {i}/{aantal}', end='\r')
    print(f'    {aantal}/{aantal}')

    return filtered_data, filtered_label

def filter_split(data, labels, model, size):
    aantal = len(data)
    filtered_data = []
    filtered_label = []
    result = []

    for i in range(aantal):
        image = data[i]
        M = image.shape[0]//2
        N = image.shape[1]//2
        tiles = [image[y:y+N, x:x+M] for y in range(0,image.shape[1],N) for x in range(0,image.shape[0],M)]
        x_center, y_center = labels[i]
        # print(x_center, y_center)
        for j in range(4):
            img = tf.reshape(tiles[j], [1, size, size, 3])
            prediction = model.predict(img, verbose=0)
            # print(j%2, j//2)
            coordinate = [x_center - size/2 + size*(j%2) , y_center - size/2 + size*(j//2)]
            # print(coordinate)
            # print(prediction)
            if prediction[0] > 0.5:
                filtered_data.append(tiles[j])
                filtered_label.append(coordinate)
                continue
            result.append(prediction[0])
        
        print(f"    {i}/{aantal}", end='\r')
    print(f"    {aantal}/{aantal}")

    return filtered_data, filtered_label, result

path = './testing/raw'
filtered_path = './testing/filtered/'
function_path = './testing/function/'
teller = 0
numfiles = len(os.listdir(path))
existing = os.listdir(function_path)

for file in os.listdir(path):
    teller += 1
    name = file.split('.')[0]
    print(f'{teller}/{numfiles}: {name}')

    if f'{name}.png' in existing:
        continue

    image = cv2.imread(f'{path}/{name}.JPG')
    labels, pieces = img_to_512(image)

    print('  model512')
    model512 = joblib.load('./trained_models/kmeans')
    filtered_data, filtered_label = filter_512(pieces, labels, model512)

    filtered_data = np.array(filtered_data)
    filtered_data = filtered_data / 255

    print('  model256')
    model256 = keras.models.load_model('./trained_models/model256_classweight/')
    filtered_data256, filtered_label256, result = filter_split(filtered_data, filtered_label, model256, 256)

    print('  model128')
    model128 = keras.models.load_model('./trained_models/model128/')
    filtered_data128, filtered_label128, result = filter_split(filtered_data256, filtered_label256, model128, 128)

    print('  model64')
    model64 = keras.models.load_model('./trained_models/model64/')
    filtered_data64, filtered_label64, result = filter_split(filtered_data128, filtered_label128, model64, 64)

    aantal = len(filtered_data64)
    for i in range(aantal):
        x, y = filtered_label64[i]
        skimage.io.imsave('{:s}/{:s}_{:.0f}_{:.0f}.png'.format(filtered_path,name,x,y), image[int(y)-32:int(y)+32, int(x)-32:int(x)+32])

    ax = plt.subplot()
    ax.imshow(image)
    ax.set_title(f'prediction {name}')
    ax.axis('off')

    size = 512
    for x, y in filtered_label:
        rect = patches.Rectangle([x-size/2, y-size/2], size, size, edgecolor = 'aqua', facecolor= 'aqua', alpha=0.25)
        ax.add_patch(rect)
    size = 256
    for x, y in filtered_label256:
        rect = patches.Rectangle([x-size/2, y-size/2], size, size, edgecolor = 'yellow', facecolor= 'yellow', alpha=0.25)
        ax.add_patch(rect)
    size = 128
    for x, y in filtered_label128:
        rect = patches.Rectangle([x-size/2, y-size/2], size, size, edgecolor = 'orange', facecolor= 'orange', alpha=0.5)
        ax.add_patch(rect)
    size = 64
    for x, y in filtered_label64:
        rect = patches.Rectangle([x-size/2, y-size/2], size, size, edgecolor = 'darkred', facecolor= 'darkred', alpha=0.5)
        ax.add_patch(rect)
    plt.savefig(f'{function_path}/{name}.png')
    plt.cla()
    
    if teller == 2:
        break


