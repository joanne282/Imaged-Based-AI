import cv2 as cv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(color=(128,128,128), thickness=1, circle_radius=1)
cap = cv.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
    cv.imshow('MediaPipe FaceMesh', image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
face_mesh.close()
cap.release()
