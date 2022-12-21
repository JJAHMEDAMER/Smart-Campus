import cv2
import time

'''
    1. Get Image or video capture
    2. Load detection algorithm
    3. transform image to gray scale
    4. detect the faces on the gray image
    5. The detection return the coordinate around the face
    6. Draw a Rectangle on the colored image
'''

start_time = time.time()

webcam = cv2.VideoCapture(0)
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ignore, frame = webcam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(gray_frame,1.05,5)  # return a list of face coordinates

    for x, y, w, h in faces:  # We loop through a list if there are multiple faces
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 10)

    cv2.imshow("Webcam", frame)

    # print (cv2.waitKey(1))
    if cv2.waitKey(1) == 32: # Space Bar to close
        break
    
webcam.release
cv2.destroyAllWindows()

end_time = time.time()
print(end_time - start_time)