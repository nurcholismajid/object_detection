import numpy as np
import cv2

# data training haarscascade
face = cv2.CascadeClassifier('....') # e.g 'frontalface.xml'
# video capture: using main camera
cap = cv2.VideoCapture(0)

while(True):
    # capture frame-by-frame
    ret, frame = cap.read()

    # our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
