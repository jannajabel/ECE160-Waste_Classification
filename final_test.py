from ast import Return
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
import pyshine as ps
import cv2 as cv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
from pylab import *
import imutils
import numpy


new_model = tf.keras.models.load_model("D:/FILES/SCHOOL/160/ECE160-Waste_Classification/version_three.h5")

DIR = "D:/FILES/SCHOOL/160/ECE160-Waste_Classification/Dataset"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(DIR, validation_split=0.1, subset="training", seed=42, batch_size=128, smart_resize=True, image_size=(256, 256))

dic = train_dataset.class_names
numClasses = len(train_dataset.class_names)

def capture_waste():
    cam = cv2.VideoCapture(0) 
    cv2.namedWindow("Waste Classifier")
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Waste Classifier Testing", frame)
    
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "D:/FILES/SCHOOL/160/ECE160-Waste_Classification/Captured/waste_image{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()

    path = "D:/FILES/SCHOOL/160/ECE160-Waste_Classification/Captured/waste_image0.jpg"

    img = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    image = cv2.imread(path)
    output = image.copy()
    output = imutils.resize(output, width=400)

    predictions = new_model.predict(img_array)[0]
    i = np.argmax(predictions)
    waste_types = ['Aluminum','Cardboard','Carton','Glass','Organic Waste','Other Plastics','Paper','Plastic','Textiles','Wood']
    test_d = 'D:/FILES/SCHOOL/160/ECE160-Waste_Classification/Dataset' + waste_types[i] + '/'
    label = waste_types[i]

    text = "{}: {:.2f}%".format(label, predictions[i]* 100)
    images =  ps.putBText(image,text,text_offset_x = 110,
                        text_offset_y = 400,
                        vspace = 10,
                        hspace = 10,
                        font_scale = 1,
                        background_RGB = (255,225,255),
                        text_RGB = (0,0,0),
                        thickness = 3,
                        alpha = 0.5
                        )

    plt.imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB), interpolation = 'bicubic')
    plt.savefig('D:/FILES/SCHOOL/160/ECE160-Waste_Classification/Waste Type/Test{0}.jpg'.format(i))
    plt.show()
    print(predictions[0]*100, "\n", dic)
    capture_waste()
    
capture_waste()


