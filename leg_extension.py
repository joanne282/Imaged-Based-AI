import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('')

with mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        result = img.copy()

        # mediapipe는 RGB를 사용하고, openCV는 BGR을 사용함 -> 따라서 두번의 변환 작업이 필요함
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            hip_y = int(results.pose_landmarks.landmark[24].y * img.shape[0])

            leg_img = img[hip_y:]
            leg_img_ori = leg_img.copy()
            leg_img = cv2.resize(leg_img, dsize=None, fx=1.0, fy=1.25)
            leg_img = leg_img[:int(leg_img.shape[0] * 0.8)]
            leg_img = cv2.resize(leg_img, dsize=(leg_img_ori.shape[1],leg_img_ori.shape[0]))

            result[hit_y] = leg_img
        
        # 선까지 출력하기
        # mp_drawing.draw_landmarks(
        #     img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS )

        cv2.imshow('leg_img_ori', leg_img_ori)
        cv2.imshow('leg_img', leg_img)
        cv2.imshow('img', img)
        cv2.imshow('result', result)

        if cv2.waitKey(1) == ord('q'):
            break
