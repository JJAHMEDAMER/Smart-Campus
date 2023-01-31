# Import packages
import cv2
import matplotlib.pyplot as plt

face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('test_images/1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detection.detectMultiScale(img_gray,1.05, 55)

fig, ax = plt.subplots(ncols=3)  # ax is a list of axes

fig.suptitle('Single Face Detection')

ax[0].imshow(img)
ax[0].set_title("Original")

x, y, w, h = faces[0]  # Faces is a list of list, each list has the coordinates of one face 
detected_face = cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 10)
ax[1].imshow(detected_face)
ax[1].set_title("detect")

cropped_img = img[y:y+h, x:x+w] # Y -> X
ax[2].imshow(cropped_img)
ax[2].set_title("cropped")
    

plt.show()
    
    



