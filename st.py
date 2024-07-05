import streamlit as st

import tkinter as tk
from tkinter import filedialog
from tkinter import *
import dlib
from sklearn import metrics

from tensorflow.keras.models import model_from_json,load_model
from PIL import Image, ImageTk
import numpy as np
import cv2

# donwload haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"
    # images.append(np.asarray(cv2.resize(cv2.imread(data_path+'/' + image[2] + '/' + image[1], cv2.IMREAD_COLOR), img_size[0:2])[:, :, ::-1]))
detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r" ) as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam',  loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

top =tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
model = FacialExpressionModel('dogemotion.json','dogemotion.h5')



EMOTIONS_LIST = ["Angry","Happy","Sad","Relaxed"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    
    dets = detector(gray_image, upsample_num_times=1)
    img_result = gray_image.copy()

   
    print(dets)
    img_size = (192, 192, 3)
    try:
        for (x,y,w,h) in dets:
            x1, y1 = x, x+w
            x2, y2 = y, y+h

            fc = cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255,0,0), lineType=cv2.LINE_AA)
            
            # fc = gray_image[y:y+h,x:x+w]
            print(fc)
            roi = np.asarray(cv2.resize(fc,img_size[0:2])[:, :, ::-1])
            print(roi)
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
        print("Predicted Emotion is" + pred)
        label1.configure(foreground="#011638",text = pred)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")


def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except: 
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Emotion Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()
top.mainloop()