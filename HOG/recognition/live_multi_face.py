import cv2
import face_recognition
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import ravel
from math import ceil

known_image_dir = 'test_images\\rec_our_images'
known_img_list = os.listdir(known_image_dir)

known_img_encoding = []
known_img_names = []
color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

# known Image
for img in known_img_list:
    img_path = os.path.join(known_image_dir, img)
    
    x = face_recognition.load_image_file(img_path)
    x_encoding = face_recognition.face_encodings(x)[0]
    
    known_img_encoding.append(x_encoding)
    known_img_names.append(img[:-4])

video_capture  = cv2.VideoCapture(0)
while True:
    name = 'Unknown'
    color = (0, 0, 0)

    ret, frame = video_capture.read() 
    # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
    # known Image
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_img_encoding, face_encoding)
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_img_encoding, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_img_names[best_match_index]
            color = color_list[best_match_index]
            
        frame = cv2.rectangle(frame, (right, top), (left, bottom), color, 3)
        # face_image = img[top:bottom, left:right]  # crop
        
        # Label
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1, (255, 255, 255), 2)
    
    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
