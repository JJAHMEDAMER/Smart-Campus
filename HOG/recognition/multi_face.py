import cv2
import face_recognition
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import ravel
from math import ceil 

known_image = 'test_images\\rec_known_images\\Elon-Musk.jpg'

unknown_image_dir = 'test_images\\rec_test_images'
img_list = os.listdir(unknown_image_dir)

# known Image
elon = face_recognition.load_image_file(known_image)
elon_encoding = face_recognition.face_encodings(elon)[0]


fig, axs = plt.subplots(ncols=(ceil(len(img_list)/3)), nrows=3)
flat_axs = ravel(axs)

i = 0
for img in img_list:
    img = os.path.join(unknown_image_dir, img)
    
    # known Image
    img = face_recognition.load_image_file(img)
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    name = 'elon musk'

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces([elon_encoding], face_encoding)
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance([elon_encoding], face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            img = cv2.rectangle(img, (right, top), (left, bottom), (255, 0, 0), 20)
            # face_image = img[top:bottom, left:right]  # crop
            
            # Label
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)
        else:
            img = cv2.rectangle(img, (right, top), (left, bottom), (0, 0, 255), 20)
            # face_image = img[top:bottom, left:right]  # crop
            
            # Label
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, f"not {name}", (left + 6, bottom - 6), font, 2.0, (255, 255, 255), 2)
            
    flat_axs[i].imshow(img)
    i += 1

plt.show()
