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


DIR = "/content/ECE160-Waste_Classification/Dataset"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(DIR, validation_split=0.1, subset="training", seed=42, batch_size=128, smart_resize=True, image_size=(256, 256))
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(DIR, validation_split=0.1, subset="validation", seed=42, batch_size=128, smart_resize=True, image_size=(256, 256))

classes = train_dataset.class_names
numClasses = len(train_dataset.class_names)
print(classes)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

new_model = tf.keras.models.load_model("/content/drive/MyDrive/version_200epochs.h5")
new_model.evaluate(test_dataset)

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

  from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

DIR = "/content/ECE160-Waste_Classification/Dataset"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(DIR, validation_split=0.1, subset="training", seed=42, batch_size=128, smart_resize=True, image_size=(256, 256))

dic = train_dataset.class_names
numClasses = len(train_dataset.class_names)


path = "/content/photo.jpg"

img = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 
image = cv2.imread(path)
output = image.copy()
output = imutils.resize(output, width=400)


predictions = new_model.predict(img_array)[0]
i = np.argmax(predictions)
waste_types = ['Aluminum','Cardboard','Carton','Glass','Organic Waste','Other Plastics','Paper','Paper','Plastic','Textiles','Wood']
test_d = '/content/ECE160-Waste_Classification/Dataset' + waste_types[i] + '/'
label = waste_types[i]



text = "{}: {:.2f}%".format(label, predictions[i]* 100)
images =  ps.putBText(image,text,text_offset_x = 110,
                      text_offset_y = 400,
                      vspace = 10,
                      hspace = 10,
                      font_scale = 1.5,
                      background_RGB = (255,225,255),
                      text_RGB = (0,0,0),
                      thickness = 3,
                      alpha = 0.5
                      )

plt.imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB), interpolation = 'bicubic')
print(predictions[0]*100, "\n", classes)