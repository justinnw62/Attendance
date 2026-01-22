import cv2
import dlib
import face_recognition
import numpy as np
print("✅ All imports successful!")

# Test webcam access
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Webcam access successful!")
    cap.release()
else:
    print("❌ Could not access webcam")