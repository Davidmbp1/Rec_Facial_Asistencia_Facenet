import cv2
import os
import joblib
import numpy as np
import time
from PIL import Image
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#Si no hay carpetas para guardar las img de train y validation las creamos
try:os.makedirs('faces')
except:pass

try:os.makedirs('faces/train')
except:pass

try:os.makedirs('faces/val')
except:pass


#Escribe tu nombre
nombre = input('Enter your name --> ')


#Verficar si algun usuario ya esta registrado
if nombre in os.listdir('faces/train'):
    print('Este usuario ya está registrado')

else:    
    os.makedirs('faces/train/'+nombre)
    os.makedirs('faces/val/'+nombre)

    cap = cv2.VideoCapture(0)
    i = 0
    print()
    for i in range(5):
        print(f'Capturing starts in {5-i} seconds...')
        time.sleep(1)
    print('Tomando fotos, sonría :) ...')
    while i<=500:
        ret,frame = cap.read()
        while ret == False:
            print("Can't receive frame. Retrying ...")
            cap.release()
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
        cv2.imshow('taking your pictures',frame)
        if i%5==0 and i<=300 and i!=0:
            cv2.imwrite('faces/train/'+nombre+'/'+str(i)+'.png',frame)
        elif i%5==0 and i>300:
            cv2.imwrite('faces/val/'+nombre+'/'+str(i)+'.png',frame)
        i+=1
            
        if cv2.waitKey(1)==27:  #Escape Key
            break

    cv2.destroyAllWindows()
    cap.release()
    print('Successfully taken your photos...')