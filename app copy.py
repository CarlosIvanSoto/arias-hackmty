from turtle import delay
import cv2
import os
import imutils
import entrenando
import reconocimientoEmociones

def crear(emotionName):
    #emotionName = 'Enojo'
    #emotionName = 'Felicidad'
    #emotionName = 'Sorpresa'
    #emotionName = 'Tristeza'

    dataPath = 'C:/Users/Manuel Sandoval/source/estadosDeAnimo/Data' #Cambia a la ruta donde hayas almacenado Data
    emotionsPath = dataPath + '/' + emotionName

    if not os.path.exists(emotionsPath):
        print('Carpeta creada: ',emotionsPath)
        os.makedirs(emotionsPath)

        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        count = 0

        while True:

            ret, frame = cap.read()
            if ret == False: break
            frame =  imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()

            faces = faceClassif.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                
                cv2.putText(frame, 'Has cara de '+emotionName,(x-30,y-30),2,0.8,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(emotionsPath + '/rotro_{}.jpg'.format(count),rostro)
                count = count + 1
            cv2.imshow('frame',frame)

            k =  cv2.waitKey(1)
            if k == 27 or count >= 200:
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    crear('Enojo')
    crear('Felicidad')
    crear('Sorpresa')
    crear('Tristeza')

        
    #entrenando.obtenerModelo('EigenFaces')
    entrenando.obtenerModelo('FisherFaces')
    entrenando.obtenerModelo('LBPH')
    reconocimientoEmociones.openCam()

if __name__ == "__main__":
    main()