import cv2
import face_recognition
import os
import matplotlib.pyplot as plt
import numpy as np

known_image = 'test_images\\1.jpg'

unknown_image_dir = 'test_images\\elon_musk\\elon_mask_1.jpg'

fig, axs = plt.subplots()

# known Image
elon = face_recognition.load_image_file(known_image)
elon_encoding = face_recognition.face_encodings(elon)[0]

# known Image
img = face_recognition.load_image_file(unknown_image_dir)
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

axs.imshow(img)

plt.show()
