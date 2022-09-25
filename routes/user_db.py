from turtle import delay
import os
import cv2
import imutils
import numpy as np
import time


camera = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def registerUser(username):
    dataPath = "data"
    userPath = dataPath + "/" + username
    if not os.path.exists(userPath):
        os.makedirs(userPath)
        
    count = 0
    while True:
        ret, frame = camera.read()
        if ret == False: 
            break
        frame =  imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = face_detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            #cv2.putText(frame, 'Has cara de '+emotionName,(x-30,y-30),2,0.8,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(720,720),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(userPath + '/rostro_{}.jpg'.format(count),rostro)
            count = count + 1
        cv2.imshow('frame',frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        if count >= 200:
            obtenerModelo(username)
            break
            #auth(camera, faceClassif)

def obtenerModelo(username):
    dataPath = "xml"
    modelPath = dataPath + "/" + username + "/"
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    # Almacenando el modelo obtenido
    #emotion_recognizer.write("modelo"+method+".xml")

    dataPath = 'data' #Cambia a la ruta donde hayas almacenado Data
    usersList = os.listdir(dataPath)
    print('Lista de usuarios: ', usersList)

    labels = []
    facesData = []
    label = 0

    for usernameDir in usersList:
        userPath = dataPath + '/' + usernameDir

        for fileName in os.listdir(userPath):
            #print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(userPath+'/'+fileName,0))
            #image = cv2.imread(emotionsPath+'/'+fileName,0)
            #cv2.imshow('image',image)
            #cv2.waitKey(10)
        label = label + 1
    
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    # Entrenando el reconocedor de rostros
    print("Entrenando ( EigenFace )...")
    inicio = time.time()
    face_recognizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time()-inicio
    print("Tiempo de entrenamiento ( EigenFace ): ", tiempoEntrenamiento)
    
    face_recognizer.write(modelPath + "modeloEigenFace.xml")

def auth():
    dataPath = "data"
    imagePaths = os.listdir(dataPath)
    
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    while True:
        ret,frame = camera.read()
        if ret == False: 
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        faces = face_detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

            # EigenFaces
            if result[1] < 5800:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('frame',frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
            

"""
if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    register(camera,faceClassif,"manuel")
"""