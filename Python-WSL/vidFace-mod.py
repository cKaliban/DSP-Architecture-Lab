import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Assume that the face coordinate is in the middle of detected face
        coord_X ="X = " + str(x + w/2)
        coord_Y ="Y = " + str(y + h/2)
        cv2.putText(img, coord_X, (x+w+10, y+h-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
        cv2.putText(img, coord_Y, (x+w+10, y+h), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
        # cv2.putText(img, "X", (x+w//2-1, y+h//2-1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()