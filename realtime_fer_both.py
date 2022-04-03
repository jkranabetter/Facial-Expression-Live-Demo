import cv2, sys, numpy as np, os
import _pickle as cPickle
from skimage.feature import hog
import dlib
from tensorflow import keras

def fetch_landmarks(image, rects, predictor):
        if len(rects) > 1:
            raise BaseException('too many faces')
        if len(rects) == 0:
            raise BaseException('no faces')
        return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)\
svm_emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']
neural_emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_DUPLEX

# load svm model
if os.path.isfile('svm_model.bin'):
    with open('svm_model.bin', 'rb') as f:
        svm_model = cPickle.load(f)

# load neural model
neural_model = keras.models.load_model("./model_FER2013_7_mobileNet")

# load Dlib predictor for face landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

webcam = cv2.VideoCapture(0)
#webcam.open(0, cv2.CAP_DSHOW)

neural_emotion = ''
svm_emotion = ''

while (webcam.isOpened()):
    ret, frame = webcam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (48,48), interpolation = cv2.INTER_AREA)
    
        # neural stuff
        preds = np.zeros(shape=(1,48,48,3))
        img2 = np.stack((face_crop,)*3, axis=-1) 
        preds[0] = img2
        preds = preds / 255.
        neural_results = neural_model.predict(preds)[0]
        neural_max_value = max(neural_results)

        # svm stuff
        hog_features, hog_image = hog(face_crop, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        image_int8 = np.uint8(face_crop)
        face_rectangles = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = fetch_landmarks(image_int8, face_rectangles, predictor)
        face_landmarks = face_landmarks.getA1()
        datas = np.concatenate((face_landmarks, hog_features), axis=None)
        svm_results = svm_model.predict([datas])
        svm_probabilities = svm_model.predict_proba([datas])
        svm_probabilities = svm_probabilities[0] #unwrap
        svm_max_value = max(svm_probabilities)


        # concatenate the images
        image_width = len(image[0])
        image_height = len(image)
        image = cv2.vconcat([image, image])
        
        # update svm labels
        if svm_max_value > 0.5:
            svm_emotion = svm_emotions[svm_results[0]]
            print("SVM: " + svm_emotion + " confidence level of " + str(svm_max_value*100) + "%")

        # update neural labels
        if neural_max_value > 0.5:
            neural_emotion = neural_emotions[np.where(neural_results==neural_max_value)[0][0]]
            print("Neural: " + neural_emotion + " confidence level of " + str(neural_max_value*100) + "%")

        # text on image
        cv2.putText(image, svm_emotion, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(image, neural_emotion, (x + 6 , y - 6 + image_height), font, 1.0, (255, 255, 255), 1)
        cv2.putText(image, 'Support Vector', (0, 25), font, 1.0, (255, 255, 255), 1)
        cv2.putText(image, 'Neural Network', (0, image_height + 25), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('FER',image)
        cv2.waitKey(25)


# After the loop release the cap object
webcam.release()
# Destroy all the windows
cv2.destroyAllWindows()