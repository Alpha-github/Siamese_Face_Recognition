import numpy as np
from numpy import expand_dims
from matplotlib import pyplot as plt
from detect_faces import *
from keras_facenet import FaceNet
import csv
import cv2
from pathlib import Path

embedder = FaceNet()

headers = ['filename', 'embedding']


embd = {}
paths = list(map(str,Path('.\\raw_dataset').rglob('*.jpg')))
# filenames = [i.stem for i in list(Path('.\\raw_dataset').rglob('*.jpg'))]
for pth in paths:
    frame = cv2.imread(pth)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    # image_copy = np.copy(mp_image.numpy_view())
    annotated_image,cropped_list,startpts_list = visualize(mp_image.numpy_view(), detection_result)
    fname = Path(pth).stem
    for cropped_img in cropped_list:
        cropped_img = cv2.resize(cropped_img, (160,160))
        cropped_img = expand_dims(cropped_img,axis=0)
        emb = list(embedder.embeddings(cropped_img)[0])
        
        if embd.get(pth) is None:
            embd[fname] = emb

print(len(embd))
with open('.\\raw_dataset\\embedding_list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for key, value in embd.items():
        writer.writerow([key, value])