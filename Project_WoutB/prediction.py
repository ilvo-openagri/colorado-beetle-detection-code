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
import seaborn as sn
import pandas as pd
import time

#variabels for time and confusion matrices
stats = False
time512 = 0
time256 = 0
time128 = 0
time64 = 0
array256 = [[0,0],[0,0]]
array128 = [[0,0],[0,0]] 
array64 = [[0,0],[0,0]]
resize_time = 0

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
        start_resize = time.time()
        img = skimage.transform.resize(data[i], (64,64))
        img = img.reshape((1,-1))
        if stats:
            global resize_time
            resize_time += time.time() - start_resize
        if model.predict(img) == 1:
            filtered_data.append(data[i])
            filtered_label.append(labels[i])
        print(f'\t{i}/{aantal}', end='\r', flush=True)
    print(f'\t{aantal}/{aantal}', flush=True)
    
    return filtered_data, filtered_label

def filter_split(data, labels, model, size):
    aantal = len(data)
    filtered_data = []
    filtered_label = []

    for i in range(aantal):
        image = data[i]
        M = image.shape[0]//2
        N = image.shape[1]//2
        tiles = [image[y:y+N, x:x+M] for y in range(0,image.shape[1],N) for x in range(0,image.shape[0],M)]
        x_center, y_center = labels[i]
        # print(x_center, y_center)
        for j in range(4):
            tile = skimage.transform.resize(tiles[j], (size,size))
            prediction = model.predict(tf.reshape(tile, [1, size, size, 3]), verbose=0)
            # print(j%2, j//2)
            coordinate = [x_center - size/2 + size*(j%2) , y_center - size/2 + size*(j//2)]
            # print(coordinate)
            # print(prediction)
            if prediction[0] > 0.5:
                filtered_data.append(tiles[j])
                filtered_label.append(coordinate)
                continue
        
        print(f"\t{i}/{aantal}", end='\r', flush=True)
    print(f"\t{aantal}/{aantal}", flush=True)

    return filtered_data, filtered_label

def test_model(data, predict_label, true_labels, size):
    try:
        lines = [line.rstrip() for line in true_labels.readlines()]
    except:
        lines = []

    aantal = len(predict_label)
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(predict_label)):
        delete = []
        x, y = predict_label[i]
        for line in lines:
            _, x_center, y_center, _, _ = [float(s) for s in line.split("\t")]
            if (x-size/2 < x_center < x+size/2) and (y-size/2 < y_center < y+size/2):
                TP += 1
                delete.append(line)
        if len(delete):
            for el in delete:
                lines.remove(el)

        FP = aantal - TP if aantal > TP else 0

    FN = len(lines)
    TN = len(data)*4 - FN - TP - FP
    
    array = [[TN, FP],
            [FN, TP]]

    return array

def plot_confusion_matrices(array256, array128, array64):
    f1_256 = (2*array256[1][1])/(2*array256[1][1]+array256[0][1]+array256[1][0])
    f1_128 = (2*array128[1][1])/(2*array128[1][1]+array128[0][1]+array128[1][0])
    f1_64 = (2*array64[1][1])/(2*array64[1][1]+array64[0][1]+array64[1][0])

    fig, ax = plt.subplots(nrows=2, ncols=2)
    df256 = pd.DataFrame(array256, range(2), range(2))
    sn.heatmap(df256, ax=ax[0][0], annot=True, center=array256[0][0], fmt='g', cbar=False)
    ax[0][0].set_title('256')
    df128 = pd.DataFrame(array128, range(2), range(2))
    sn.heatmap(df128, ax=ax[0][1], annot=True, center=array128[0][0], fmt='g', cbar=False)
    ax[0][1].set_title('128')
    df64 = pd.DataFrame(array64, range(2), range(2))
    sn.heatmap(df64, ax=ax[1][0], annot=True, center=array64[0][0], fmt='g', cbar=False)
    ax[1][0].set_title('64')

    ax[1][1].set_visible(False)

    fig.text(0.55,0.3,f'f1-score 256: {round(f1_256,4)}\nf1-score 128: {round(f1_128,4)}\nf1-score 64: {round(f1_64,4)}')

    plt.savefig('./documentation/test.png')

def predict_image(image, name):
    labels, pieces = img_to_512(image)

    if stats:
        start512 = time.time()
    print('    model512', flush=True)
    model512 = joblib.load('./trained_models/kmeans')
    filtered_data, filtered_label = filter_512(pieces, labels, model512)

    filtered_data = np.array(filtered_data)
    filtered_data = filtered_data / 255

    if stats:
        global time512
        time512 += (time.time() - start512) #/ len(pieces)
        start256 = time.time()
    
    print('    model256', flush=True)
    model256 = keras.models.load_model('./trained_models/model256_classweight/')
    filtered_data, filtered_label = filter_split(filtered_data, filtered_label, model256, 256)

    if stats:
        labels = open(f"./data/labels_full_txt/{name}.txt")
        global array256
        array256 = test_model(filtered_data, filtered_label, labels, 256)
        global time256
        time256 += (time.time() - start256) #/ len(filtered_data)
        start128 = time.time()

    print('    model128', flush=True)
    model128 = keras.models.load_model('./trained_models/model128/')
    filtered_data, filtered_label = filter_split(filtered_data, filtered_label, model128, 128)

    if stats:
        labels = open(f"./data/labels_full_txt/{name}.txt")
        global array128
        array128 = test_model(filtered_data, filtered_label, labels, 128)
        global time128
        time128 += (time.time() - start128) #/ len(filtered_data)
        start64 = time.time()

    print('    model64', flush=True)
    model64 = keras.models.load_model('./trained_models/model64/')
    filtered_data, filtered_label = filter_split(filtered_data, filtered_label, model64, 64)

    if stats:
        labels = open(f"./data/labels_full_txt/{name}.txt")
        global array64
        array64 = test_model(filtered_data, filtered_label, labels, 64)
        global time64
        time64 += (time.time() - start64) #/ len(filtered_data)

    return filtered_label

def performantie(image, name):
    labels, pieces = img_to_512(image)

    start512 = time.time()
    print('    model512', flush=True)
    model512 = joblib.load('./trained_models/kmeans')
    filtered_data, filtered_label = filter_512(pieces, labels, model512)
    filtered_data = np.array(filtered_data)
    filtered_data = filtered_data / 255
    global time512
    time512 += (time.time() - start512) #/ len(pieces)
    
    start256 = time.time()
    print('    model256', flush=True)
    model256 = keras.models.load_model('./trained_models/model256_classweight/')
    _, _ = filter_split(filtered_data, filtered_label, model256, 256)
    global time256
    time256 += (time.time() - start256) #/ len(filtered_data)
    
    start128 = time.time()
    print('    model128', flush=True)
    model128 = keras.models.load_model('./trained_models/model128/')
    _, _ = filter_split(filtered_data, filtered_label, model128, 128)
    global time128
    time128 += (time.time() - start128) #/ len(filtered_data)
    
    start64 = time.time()
    print('    model64', flush=True)
    model64 = keras.models.load_model('./trained_models/model64/')
    _, _ = filter_split(filtered_data, filtered_label, model64, 64)
    global time64
    time64 += (time.time() - start64) #/ len(filtered_data)

    return filtered_label

def main(path, statictics=False):
    '''
        Test, dit is de documentatie.
    '''
    global stats
    stats = statictics
    answer = {}
    start_time = time.time()

    aantal = len(os.listdir(path))
    teller = 0

    for file in os.listdir(path):
        teller += 1

        print(f'{teller}/{aantal}', flush=True)
        image = skimage.io.imread(f'{path}/{file}')
        filename = file.split('.')[0]

        # labels = predict_image(image, filename)
        labels = performantie(image, filename)

        # for i in range(2):
        #     for j in range(2):
        #         array256[i][j] = array256[i][j] + new_array256[i][j]
        #         array128[i][j] = array128[i][j] + new_array128[i][j]
        #         array64[i][j] = array64[i][j] + new_array64[i][j]

        answer[file] = labels
        

    if stats:
        # plot_confusion_matrices(array256, array128, array64)
        print(f'''Time per piece
512: {round(time512,3)}s
resize_time: {round(resize_time,3)}s
256: {round(time256,3)}s
128: {round(time128,3)}s
64:  {round(time64,3)}s
            ''')
    
    total_time = time.time() - start_time
    print(f'Time elapsed: {total_time//60}m {round(total_time%60,3)}s')
    return answer



path = './data/images_test'

labels = main(path, True)
# print([el for el in labels])

