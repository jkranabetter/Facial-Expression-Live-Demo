import cv2, sys, numpy as np, os
from tensorflow import keras
import copy
from PIL import ImageGrab

'''
Emotion Recognition
Run neural prediction on screen capture

Joshua Kranabetter and Taif Anjum
2022
'''

model = keras.models.load_model("./model_FER2013_7_mobileNet")
emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]
#emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_DUPLEX

while (True):
    img = ImageGrab.grab(bbox=(0, 0, 1100, 1100))
    img = np.array(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (48,48), interpolation = cv2.INTER_AREA)

        preds = np.zeros(shape=(1,48,48,3))
        img2 = np.stack((face_crop,)*3, axis=-1) 
        preds[0] = img2
        preds = preds / 255.
      
        results = model.predict(preds)[0]
        max_value = max(results)
        
        if max_value > 0.3:
            emotion = emotions[np.where(results==max_value)[0][0]]
            print(emotion + " confidence level of " + str(max_value*100) + "%")
            cv2.putText(img, emotion, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('FER',img)
    cv2.waitKey(25)

# After the loop release the cap object
#webcam.release()
# Destroy all the windows
#cv2.destroyAllWindows()