import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
cap = cv2.VideoCapture(0)
base_options = python.BaseOptions(model_asset_path=r'D:\PythonDay1\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = detector.detect(mp_image)
    if results.hand_landmarks:
        for hand_lms in results.hand_landmarks:
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
    cv2.imshow("ok", img)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty("ok", cv2.WND_PROP_VISIBLE) < 1):
        break
cap.release()
cv2.destroyAllWindows()