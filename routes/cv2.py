from flask import Blueprint, render_template, Response
import cv2
import mediapipe as mp
import os
import imutils
import numpy as np
import time
from math import acos, degrees

app = Blueprint('routes-cv2', __name__)


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
#camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
    
    face_recognizer.write(modelPath + "modeloEigenFace.xml" )

def normal():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("frame", frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
 
def auth():
    with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
        while True:
            ret, frame = camera.read()
            if ret == False:
                break
            frame = cv2.flip(frame,1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))
            cv2.imshow("frame", frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
# Pulgar
thumb_points = [1, 2, 4]
# Índice, medio, anular y meñique
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]
# Colores
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)

def handd():
    with mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            fingers_counter = "_"
            thickness = [2, 2, 2, 2, 2]
            if results.multi_hand_landmarks:
                coordinates_thumb = []
                coordinates_palm = []
                coordinates_ft = []
                coordinates_fb = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for index in thumb_points:
                        x = int(hand_landmarks.landmark[index].x * width)
                        y = int(hand_landmarks.landmark[index].y * height)
                        coordinates_thumb.append([x, y])

                    for index in palm_points:
                        x = int(hand_landmarks.landmark[index].x * width)
                        y = int(hand_landmarks.landmark[index].y * height)
                        coordinates_palm.append([x, y])

                    for index in fingertips_points:
                        x = int(hand_landmarks.landmark[index].x * width)
                        y = int(hand_landmarks.landmark[index].y * height)
                        coordinates_ft.append([x, y])

                    for index in finger_base_points:
                        x = int(hand_landmarks.landmark[index].x * width)
                        y = int(hand_landmarks.landmark[index].y * height)
                        coordinates_fb.append([x, y])
                    ##########################
                    # Pulgar
                    p1 = np.array(coordinates_thumb[0])
                    p2 = np.array(coordinates_thumb[1])
                    p3 = np.array(coordinates_thumb[2])
                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)
                    # Calcular el ángulo
                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                    thumb_finger = np.array(False)
                    if angle > 150:
                        thumb_finger = np.array(True)

                    ################################
                    # Índice, medio, anular y meñique
                    nx, ny = palm_centroid(coordinates_palm)
                    cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                    coordinates_centroid = np.array([nx, ny])
                    coordinates_ft = np.array(coordinates_ft)
                    coordinates_fb = np.array(coordinates_fb)
                    # Distancias
                    d_centrid_ft = np.linalg.norm(
                        coordinates_centroid - coordinates_ft, axis=1)
                    d_centrid_fb = np.linalg.norm(
                        coordinates_centroid - coordinates_fb, axis=1)
                    dif = d_centrid_ft - d_centrid_fb
                    fingers = dif > 0
                    fingers = np.append(thumb_finger, fingers)
                    fingers_counter = str(np.count_nonzero(fingers == True))
                    for (i, finger) in enumerate(fingers):
                        if finger == True:
                            thickness[i] = -1

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            ################################
            # Visualización
            cv2.rectangle(frame, (0, 0), (80, 80), (125, 220, 0), -1)
            cv2.putText(frame, fingers_counter, (15, 65), 1, 5, (255, 255, 255), 2)
            # Pulgar
            cv2.rectangle(frame, (100, 10), (150, 60), PEACH, thickness[0])
            cv2.putText(frame, "Pulgar", (100, 80), 1, 1, (255, 255, 255), 2)
            # Índice
            cv2.rectangle(frame, (160, 10), (210, 60), PURPLE, thickness[1])
            cv2.putText(frame, "Indice", (160, 80), 1, 1, (255, 255, 255), 2)
            # Medio
            cv2.rectangle(frame, (220, 10), (270, 60), YELLOW, thickness[2])
            cv2.putText(frame, "Medio", (220, 80), 1, 1, (255, 255, 255), 2)
            # Anular
            cv2.rectangle(frame, (280, 10), (330, 60), GREEN, thickness[3])
            cv2.putText(frame, "Anular", (280, 80), 1, 1, (255, 255, 255), 2)
            # Menique
            cv2.rectangle(frame, (340, 10), (390, 60), BLUE, thickness[4])
            cv2.putText(frame, "Menique", (340, 80), 1, 1, (255, 255, 255), 2)

            cv2.imshow('frame',frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
            
def emotionImage(emotion):
    # Emojis
    if emotion == 'Felicidad': image = cv2.imread('avatares/felicidad.jpg')
    if emotion == 'Enojo': image = cv2.imread('avatares/enojo.jpg')
    if emotion == 'Sorpresa': image = cv2.imread('avatares/sorpresa.jpg')
    if emotion == 'Tristeza': image = cv2.imread('avatares/tristeza.jpg')
    return image


def avatar():
    """
    method = 'LBPH'
    if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH': """
    emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
    emotion_recognizer.read('modeloLBPH.xml')
    # --------------------------------------------------------------------------------
    dataPath = 'data' #Cambia a la ruta donde hayas almacenado Data
    imagePaths = os.listdir(dataPath)
    print('imagePaths=',imagePaths)

    cap = cv2.VideoCapture(0)

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    while True:

        ret,frame = cap.read()
        if ret == False: 
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        #nFrame = cv2.hconcat([frame, np.zeros((480 ,300,3),dtype=np.uint8)])

        faces = faceClassif.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = emotion_recognizer.predict(rostro)

            cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            # LBPHFace
            if result[1] < 60:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                image = emotionImage(imagePaths[result[0]])
                nframe = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                nframe = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        #cv2.imshow('nframe', nframe)
        
        ret, buffer = cv2.imencode('.jpg', nframe)
        if ret:
            nframe = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + nframe + b'\r\n') 
        

@app.route('/login_feed')
def login_feed():
    return Response(auth(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_feed')
def register_feed():
    return Response(normal(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/avatar_feed')
def avatar_feed():
    return Response(avatar(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/handd_feed')
def handd_feed():
    return Response(handd(), mimetype='multipart/x-mixed-replace; boundary=frame')