import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import tarfile
import random
from tensorflow import keras
from models.model import model_net
import config
import cv2
from models import model
from importlib import reload
from config import net1_config
from hyperparams import net1_hyperparams
from data import Data
from logger import Logger
from trainer import Trainer
import tensorflow as tf
from data_loader import load_fer_dataset, show_random_sample, split_dataset, emo_to_string

from os import listdir
from datetime import datetime

def main():
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)
    face_cas = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    # faces,emotions = load_fer_dataset('./fer2013/fer2013.csv')
    # train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = split_dataset(faces, emotions)
    mod = model.model_net(net1_config, net1_hyperparams)
    # d = Data(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

    with tf.Session() as sess:
        mod.load(sess)
        while True:
            ret, frame = cam.read()
        
            if ret==True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #gray = cv2.flip(gray,1)
                faces = face_cas.detectMultiScale(gray, 1.3,5)
                
                for (x, y, w, h) in faces:
                    face_component = gray[y:y+h, x:x+w]
                    fc = cv2.resize(face_component, (48, 48))
                    inp = np.reshape(fc,(1,48,48,1)).astype(np.float32)
                    inp = inp/255.
                    prediction = mod.predict(sess, inp)
                    em = emo_to_string(prediction)
                    cv2.putText(frame, em, (x, y), font, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.imshow("image", frame)
                
                if cv2.waitKey(1) == 27:
                    break
            else:
                print ('Error')

        cam.release()
        cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()

def webcam():
    print("hello")
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)
    face_cas = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cam.read()
        
        if ret==True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #gray = cv2.flip(gray,1)
            faces = face_cas.detectMultiScale(gray, 1.3,5)
            
            for (x, y, w, h) in faces:
                face_component = gray[y:y+h, x:x+w]
                fc = cv2.resize(face_component, (48, 48))
                inp = np.reshape(fc,(1,48,48,1)).astype(np.float32)
                inp = inp/255.
                # prediction = model.predict_proba(inp)
                # em = emotion[np.argmax(prediction)]
                # score = np.max(prediction)
                cv2.putText(frame, "raya", (x, y), font, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.imshow("image", frame)
            
            if cv2.waitKey(1) == 27:
                break
        else:
            print ('Error')

    cam.release()
    cv2.destroyAllWindows()