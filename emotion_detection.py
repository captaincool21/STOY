import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import requests
import serial
import time

def send_to_telegram(message):

    apiToken = '6183599995:AAG-lhQ-rTnkccM4H08YrwtnPj1Q1r6VGig'
    chatID = '1750522764'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)

# load model

last_time=0
emrg_time=0

ser = serial.Serial('COM6', 9600, timeout=0)
model = load_model("best_model.h5")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
#time.sleep(5)
while True:
    
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    if len(faces_detected) == 0:
        title = "No faces found!"
        print("No faces detected")
        ser.write(b'T')
	

        # if no face is detected
        #send_to_telegram(time.perf_counter())
        if emrg_time != 0 and time.perf_counter() - emrg_time > 100:      # 100 secs
            send_to_telegram("Urgent! Kindly have a look")
            emrg_time = time.perf_counter()
            
            
        
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        if predicted_emotion == 'angry' or predicted_emotion == 'sad':
            ser.write(b'S')
            #time.sleep(3)
        else:
            ser.write(b'T')
            

        if time.perf_counter() - last_time > 120:
            send_to_telegram(predicted_emotion)
            last_time = time.perf_counter()

        # send emotion periodically let's say after every 2 mins
        # if sadness or anger play buzzer
        # if no face is detected for 5 mins, send emergency alert.

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
