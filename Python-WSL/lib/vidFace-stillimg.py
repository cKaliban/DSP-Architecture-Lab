import cv2
from time import sleep

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')

# To use a single image as input 
img = cv2.imread("testAI.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw the rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display
cv2.imshow('img', img)

k = cv2.waitKey(0)
cv2.imwrite("AI_proc.jpg",img)

