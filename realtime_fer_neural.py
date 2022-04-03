import cv2, sys, numpy as np, os
from tensorflow import keras
import copy

model = keras.models.load_model("./model_FER2013_7_mobileNet")
emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]
#emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_DUPLEX


webcam = cv2.VideoCapture(0)
#webcam.open(0, cv2.CAP_DSHOW)
while (True):
    ret, frame = webcam.read()
    #frame = copy.deepcopy(im)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (48,48), interpolation = cv2.INTER_AREA)

        preds = np.zeros(shape=(1,48,48,3))
        img2 = np.stack((face_crop,)*3, axis=-1) 
        preds[0] = img2
        print(preds)
        preds = preds / 255.
      
        results = model.predict(preds)[0]
        max_value = max(results)
        
        if max_value > 0.5:
            emotion = emotions[np.where(results==max_value)[0][0]]
            print(emotion + " confidence level of " + str(max_value*100) + "%")
            cv2.putText(image, emotion, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('FER',image)
            cv2.waitKey(25)

    
            
        

# After the loop release the cap object
#webcam.release()
# Destroy all the windows
#cv2.destroyAllWindows()