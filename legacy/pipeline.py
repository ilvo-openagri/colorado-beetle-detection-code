import numpy as np
import skimage.io
import skimage.color
import skimage.filters
import skimage.transform
import constants
import os
import time
import gc
import torch
import csv
from torchvision import transforms
import torch.nn as nn
import cv2
from random import randint
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#modelklasse
class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def split_image(img, size):
    data = []
    y = 0
    while(y < constants.Y_END - size):
        x = 0
        while(x < constants.X_END - size):
            x_end = x+size if x+size < constants.X_END else constants.X_END
            y_end = y+size if y+size < constants.Y_END else constants.Y_END
            data.append(img[y:y_end,x:x_end])
            x += size - constants.OVERLAP
        y += size - constants.OVERLAP
    return data

def downsample(img):
    downscaled = skimage.transform.resize(img, (64,64), anti_aliasing=True)
    return downscaled

def downsample_opencv(img):
    blur = cv2.GaussianBlur(img,(23,23),0)
    downscaled = cv2.resize(blur, (64,64), cv2.INTER_AREA)
    return downscaled

#functie die in vier knipt 
def cut_in_parts(data):
    new_data = []
    for im in data:
        M = im.shape[0]//2
        N = im.shape[1]//2
        tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
        for tile in tiles:
            new_data.append(tile)
    return new_data

#functie die inference doet voor een bepaald model en bepaalde data (geen datafolder) en enkel nuttige data teruggeeft
def inference(modelname, data, threshold):
    if len(data) == 0:
        return [], 0
    data_ = [downsample(np.array(i)) for i in data]
    start = time.time()
    model = Filter()
    model.cuda()
    model.load_state_dict(torch.load(f"./trained_models/{modelname}.pth"))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    input = torch.cat([transform(i).float().unsqueeze(0).to(device) for i in data_], 0)
    output = model(input)
    output = output.to(device).data.cpu().numpy()
    filtered_data = []
    for i in range(len(input)):
        if output[i] > threshold:
            filtered_data.append(data[i])
    t = time.time() - start
    return filtered_data, t

#functie die alles bij elkaar doet: data -> model1 -> filter -> model2 -> filter -> model3 -> filter -> model4; geeft patches terug die positief zijn
def predict_data(data, thresholds):
    t = 0
    all_data = []
    for i in range(1,5):
        filtered_data, t_ = inference(f"model{i}", data, thresholds[i-1])
        all_data.append(filtered_data)
        t += t_
        if(i == 4 ): break
        start = time.time()
        data = cut_in_parts(filtered_data)
        t += time.time() - start
    return filtered_data, t, all_data

def predict_data_baseline(data):
    start = time.time()
    result = inference("base_moredata", data, 0.5)
    t = time.time() - start
    return result, t

def predict_data_yolo(data):
    start = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/size512_epoch50_batch16/weights/best.pt')
    device = torch.device("cuda")
    model.to(device)
    with torch.no_grad():
        result = model(data)
        t = time.time() - start
        return result, t

def get_ground_truth(filename, count, width, x_length):
    gt = np.zeros(count)
    f = open(filename, "r")
    for line in f:
        arr = line.split("\t")
        x_center = float(arr[1])
        y_center = float(arr[2])
        index = int(x_center//width + (x_length//width) * (y_center//width))
        try:
            gt[index] = 1
        except:
            continue
    return gt

def convert_yolo_to_indexed_array(data, width, yolo_width, x_length):
    array = np.zeros(10168)
    for i in range(len(data.xyxy)):
        row = data.xyxy[i].cpu().numpy()
        for j in range(len(row)):
            det = row[j]
            x_center = det[0] + (det[2] - det[0])
            y_center = det[1] + (det[3] - det[1])
            conf = det[4]
            column_length = x_length // yolo_width
            x = x_center + (i % column_length) * yolo_width
            y = y_center + (i // column_length) * yolo_width
            #print("x: "+ str(x) + "\ty: "+ str(y))
            index = int(x//width + (x_length//width) * (y//width))
            if conf > 0.5:  
                array[index] = 1
            #print("index 512:\t" + str(i) + "\tindex 64:\t" + str(index))
    return array

def convert_result_to_indexed_array(data, start_data):
    array = np.zeros(len(start_data))
    for i in range(len(start_data)):
        for j in range(len(data)):
            if np.array_equal(start_data[i], data[j]):
                array[i] = 1
    return array

def calculate_stats(gt, predicted):
    tp = tn = fp = fn = 0
    for i in range(len(gt)):
        if gt[i] == 0 and predicted[i] == 0:
            tn += 1
        elif gt[i] == 0 and predicted[i] == 1:
            fp += 1
        elif gt[i] == 1 and predicted[i] == 0:
            fn += 1
        elif gt[i] == 1 and predicted[i] == 1:
            tp += 1
    return tp, tn, fp, fn

def calculate_metrics(tp, tn, fp, fn):
    a = (tp + tn) / (tp + tn + fp + fn)
    if tp + fp > 0:
        p = tp / (tp + fp)
    else:
        p = 0
    r = tp / (tp + fn)
    return a, p, r

#Evalueer perfomantie en snelheid van de 3 modellen
def evaluate_metrics(imagepath, labelpath):
    teller = 1
    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["y_TP","y_TN","y_FP","y_FN","y_Accuracy","y_Precision","y_Recall","y_Time", "b_TP","b_TN","b_FP","b_FN","b_Accuracy","b_Precision","b_Recall","b_Time", "p_TP","p_TN","p_FP","p_FN","p_Accuracy","p_Precision","p_Recall","p_Time"])

        for filename in os.listdir(os.path.join(os.getcwd(), imagepath)):
            print(f"{teller}/80")
            filename = filename.split(".")[0]
            img = skimage.io.imread(f"{imagepath}/{filename}.JPG")
            data_512 = split_image(img, 512)
            data_64 = split_image(img, 64)

            gt = get_ground_truth(f"{labelpath}/{filename}.txt", 10168, 64, 7952)
            gc.collect()
            torch.cuda.empty_cache()
            result_yolo, yt = predict_data_yolo(data_512)
            yolo = convert_yolo_to_indexed_array(result_yolo, 64, 512, 7952)
            ytp, ytn, yfp, yfn = calculate_stats(gt,yolo)
            ya, yp, yr = calculate_metrics(ytp, ytn, yfp, yfn)
            print("------------------------------------------")
            print("YOLO")
            print(f"tp: {ytp}\ttn: {ytn}\tfp: {yfp}\tfn: {yfn}")
            print("accuracy: "+str(ya))
            print("precision: "+str(yp))
            print("recall: "+str(yr))

            result_baseline, bt = predict_data_baseline(data_64)
            baseline = convert_result_to_indexed_array(result_baseline, data_64)
            btp, btn, bfp, bfn = calculate_stats(gt,baseline)
            ba, bp, br = calculate_metrics(btp, btn, bfp, bfn)
            print("------------------------------------------")
            print("BASELINE")
            print(f"tp: {btp}\ttn: {btn}\tfp: {bfp}\tfn: {bfn}")
            print("accuracy: "+str(ba))
            print("precision: "+str(bp))
            print("recall: "+str(br))

            result_pipeline, pt, _ = predict_data(data_512, [0.5, 0.5, 0.5, 0.5])
            pipeline = convert_result_to_indexed_array(result_pipeline, data_64)
            ptp, ptn, pfp, pfn = calculate_stats(gt,pipeline)
            pa, pp, pr = calculate_metrics(ptp, ptn, pfp, pfn)
            print("------------------------------------------")
            print("PIPELINE")
            print(f"tp: {ptp}\ttn: {ptn}\tfp: {pfp}\tfn: {pfn}")
            print("accuracy: "+str(pa))
            print("precision: "+str(pp))
            print("recall: "+str(pr))

            teller += 1

            writer.writerow([ytp, ytn, yfp, yfn, ya, yp, yr, yt, btp, btn, bfp, bfn, ba, bp, br, bt, ptp, ptn, pfp, pfn, pa, pp, pr, pt])

#Evalueer afweging van de nauwkeurigheid ten opzichte van de snelheid
def evaluate_accuracy_runtime(imagepath, labelpath):
    images_in_64 = []
    images_in_512 = []
    images_gt = []

    for filename in os.listdir(os.path.join(os.getcwd(), imagepath)):
        filename = filename.split(".")[0]
        img = skimage.io.imread(f"{imagepath}/{filename}.JPG")
        images_in_512.append(split_image(img, 512))
        images_in_64.append(split_image(img, 64))
        images_gt.append(get_ground_truth(f"{labelpath}/{filename}.txt", 10168, 64, 7952))

    with open('accuracy_runtime.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Iteration', 'Thresholds','p_Accuracy', 'p_Precision', 'p_Recall', 'p_Time'])
        for i in range(2000):
            thresholds = [randint(1,99)/100,randint(1,99)/100,randint(1,99)/100,randint(1,99)/100]
            pt_total = pa_total = pp_total = pr_total = 0
            count = len(images_gt)
            for j in range(count):
                result_pipeline, pt, _ = predict_data(images_in_512[j], thresholds)
                pipeline = convert_result_to_indexed_array(result_pipeline, images_in_64[j])
                ptp, ptn, pfp, pfn = calculate_stats(images_gt[j],pipeline)
                pa, pp, pr = calculate_metrics(ptp, ptn, pfp, pfn)
                pt_total += pt
                pa_total += pa
                pp_total += pp
                pr_total += pr
            writer.writerow([i+1, thresholds, pa_total/count, pp_total/count, pr_total/count, pt_total/count])

#Visualiseer de werking van het model, inclusief een visualisatie van de ground truth
def visualize_inference(filepath):
    #fotoplot met ground truth voor classificatie
    name = filepath.split("/")[-1].split(".")[0]
    imagename = f"./data/images_full/{name}.JPG"
    labelsname = f"./data/labels_full/{name}.txt"
    image = skimage.io.imread(imagename)
    fig, ax = plt.subplots()
    ax.imshow(image)
    gt = get_ground_truth(labelsname, 10168, 64, 7952)
    size = 64
    x = 7952
    for i in range(len(gt)):
        if gt[i] == 1:
            rect = patches.Rectangle((i%(x//size)*size, ((i*size)//x)*size), size, size, edgecolor = 'darkred', facecolor= 'darkred', alpha=0.5)
            ax.add_patch(rect)
    plt.axis('off')       
    plt.savefig(f"./data/{name}_GT.png")

    #fotoplot waarop de keuze van het model wordt geillustreerd
    image = skimage.io.imread(imagename)
    data_512 = split_image(image, 512)
    data_64 = split_image(image, 64)
    predicted, _, filtered_data_arr = predict_data(data_512, [0.5, 0.5, 0.5, 0.5])
    pipeline = convert_result_to_indexed_array(predicted, data_64)

    print(calculate_metrics(*calculate_stats(gt, pipeline)))

    fig, ax = plt.subplots()
    ax.imshow(image)
    size = 512
    x = 7952
    y = 5304

    #voor 1 laag 
    colors = ['aqua', 'yellow', 'orange', 'darkred']

    data_split = [data_512, split_image(image, 256), split_image(image, 128), data_64]

    for j in range(4):
        arr_len = len(filtered_data_arr[j])
        for i in range(len(data_split[j])):
            for k in range(arr_len):
                if np.array_equal(data_split[j][i],filtered_data_arr[j][k]):
                    rect = patches.Rectangle((i%(x//size)*size, ((i*size)//x)*size), size, size, edgecolor = colors[j], facecolor= colors[j], alpha=(j+1)/4)
                    ax.add_patch(rect)
                    break
        size /= 2
    plt.axis('off')  
    plt.savefig(f"./data/{name}_pipe.png")