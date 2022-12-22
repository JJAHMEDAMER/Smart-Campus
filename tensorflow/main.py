import cv2
import os
from time import sleep
from uuid import uuid1

# Video Stream
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    cv2.imshow('frame_vscode', frame)
    
    if cv2.waitKey(1) == ord("q"): # press q to close
        break
    
cam.release
cv2.destroyAllWindows()


# Capture images for training
IMAGES_PATH = os.path.join('data','images')
number_images = 10

cam = cv2.VideoCapture(0)

for imgnum in range(number_images):
    print(f'Collecting image {imgnum}')
    ret, frame = cam.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()