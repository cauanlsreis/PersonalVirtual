#IMPORTS
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Contador de reps
counter = 0
stage = None

#Capturando vídeo
cap = cv2.VideoCapture(0)

#Cálculo dos ângulos
def calculate_angle(a, b, c):
  a = np.array(a) #Primeiro
  b = np.array(b) #Meio
  c = np.array(c) #Final

  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
  angle = np.abs(radians*180.0/np.pi)

  if angle > 180.0:
    angle = 360 - angle
  
  return angle

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

        #Obtendo o tamanho da imagem
        image_height, image_width, _ = image.shape

        #Extraindo os pontos de referência
        try:
            landmarks = results.pose_landmarks.landmark

            #Pegando coordenadas
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            #Calculando ângulos
            angle = calculate_angle(shoulder,elbow,wrist)

            #Visualizando ângulos
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [image_width,image_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )

            #Lógica para contagem de repetições
            if angle > 160:
               stage = "down"
            if angle < 40 and stage == "down":
               stage = "up"
               counter += 1
               print(counter)

        except:
            pass
        
        #Status
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        #Contagem
        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        #Stage
        cv2.putText(image, 'STAGE', (65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

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
 