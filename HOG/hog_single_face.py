import cv2
import face_recognition
import os
import matplotlib.pyplot as plt

IMG_DIR = 'test_images'
img_path = '1.jpg'

fig, axs = plt.subplots()

# Load the jpg file into a numpy array
img_full_path = os.path.join(IMG_DIR, img_path)
img = face_recognition.load_image_file(img_full_path)
face_locations = face_recognition.face_locations(img)

for top, right, bottom, left in face_locations:
    img = cv2.rectangle(img, (right, top), (left, bottom), (255, 0, 0), 20)
    # face_image = img[top:bottom, left:right]  # crop

axs.imshow(img)
axs.set_title(img_full_path[:-4])

plt.show()
