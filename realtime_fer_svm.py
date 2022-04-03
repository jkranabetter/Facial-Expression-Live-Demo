import cv2, sys, numpy as np, os
import copy
import _pickle as cPickle
from skimage.feature import hog
import imageio
import dlib

def fetch_landmarks(image, rects, predictor):
        if len(rects) > 1:
            raise BaseException('too many faces')
        if len(rects) == 0:
            raise BaseException('no faces')
        return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)\


emotions = ['Anger', 'disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_DUPLEX

# load svm model
if os.path.isfile('svm_model.bin'):
    with open('svm_model.bin', 'rb') as f:
        model = cPickle.load(f)

# load Dlib predictor for face landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

webcam = cv2.VideoCapture(0)
#webcam.open(0, cv2.CAP_DSHOW)

last_emotion = ''

while (webcam.isOpened()):
    ret, frame = webcam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (48,48), interpolation = cv2.INTER_AREA)
    
        hog_features, hog_image = hog(face_crop, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualize=True)

        image_int8 = np.uint8(face_crop)
        face_rectangles = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = fetch_landmarks(image_int8, face_rectangles, predictor)
        face_landmarks = face_landmarks.getA1()

        datas = np.concatenate((face_landmarks, hog_features), axis=None)

        results = model.predict([datas])
        probabilities = model.predict_proba([datas])
        probabilities = probabilities[0] #unwrap

        max_value = max(probabilities)
        
        if max_value > 0.5:
            emotion = emotions[results[0]]
            last_emotion = emotion
            print(emotion + " confidence level of " + str(max_value*100) + "%")
            cv2.putText(image, emotion, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('FER',image)
            cv2.waitKey(25)
        else:
            cv2.putText(image, last_emotion, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('FER',image)
            cv2.waitKey(25)


# After the loop release the cap object
webcam.release()
# Destroy all the windows
cv2.destroyAllWindows()