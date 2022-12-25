from math import ceil
from numpy import ravel
import cv2
import face_recognition
import os
import matplotlib.pyplot as plt

IMG_DIR = 'test_images'

img_list = os.listdir(IMG_DIR)


fig, axs = plt.subplots(ncols=(ceil(len(img_list)/3)), nrows=3)

flat_axs = ravel(axs)
for i, img_path in enumerate(img_list):
    # Load the jpg file into a numpy array
    img_full_path = os.path.join(IMG_DIR, img_path)
    print(img_path)
    img = face_recognition.load_image_file(img_full_path)
    print(img.shape)
    face_locations = face_recognition.face_locations(img)
    print(face_locations)
    for top, right, bottom, left in face_locations:
        img = cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 20)
        # face_image = img[top:bottom, left:right]  # crop
        
    flat_axs[i].imshow(img)
    flat_axs[i].set_title(img_full_path[:-4])

plt.show()
# cv2.imshow("face", face_image)
    
# cv2.waitKey(0)