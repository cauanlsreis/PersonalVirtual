#IMPORTS
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#VIDEO FEED
cap = cv2.VideoCapture(0)
#Configurando instância do medipipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #Recolorindo a imagem para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Fazendo as detecções
        results = pose.process(image)

        #Recolorindo para BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Extraindo os pontos de referência
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass

        #Renderizando as detecções
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(57, 244, 1), thickness=4, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(12, 192, 9), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
