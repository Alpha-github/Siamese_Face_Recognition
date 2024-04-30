#%%
import numpy as np
from numpy import expand_dims
from numpy import asarray
from numpy import dot
from numpy.linalg import norm

from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from detect_faces import *
from keras_facenet import FaceNet
import csv
import json

#%%
attendance = set()
embedder = FaceNet()
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    print(face_array.shape)
    plt.imshow(face_array)
    return face_array

embd = {}
with open('.\\raw_dataset\\embedding_list.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        embd[row[0]] = np.array(eval(row[1]))

import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image,cropped_list,startpts_list = visualize(image_copy, detection_result)

    
    for i,cropped_img in enumerate(cropped_list):
        similarity=[]
        fnames=[]

        cropped_img = cv2.resize(cropped_img, (160,160))
        cropped_img = expand_dims(cropped_img,axis=0)
        emb = np.array(embedder.embeddings(cropped_img)[0])
        for key, value in embd.items():
            cos_sim = dot(emb, value)/(norm(emb)*norm(value))
            similarity.append(cos_sim)
            fnames.append(key)
        try:
            val = np.argmax(similarity)
            if similarity[val]>0.5 :
                attendance.add(fnames[val])
                cv2.putText(annotated_image, f'{fnames[val]}: {similarity[val]}', startpts_list[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        except:
            print('No face detected')
        
    cv2.imshow('test',annotated_image)

    if cv2.waitKey(1) == ord('q'):
            print(attendance)
            break
        
cap.release()
cv2.destroyAllWindows()
# %%
