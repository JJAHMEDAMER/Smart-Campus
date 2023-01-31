import json
import cv2
import os
from time import time

print("running...")

start = time()

IMGS_DIR = 'test_images'
img_path_list = os.listdir(IMGS_DIR)

classifiers_list = [
    'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_alt_tree.xml',
    'haarcascade_frontalface_default.xml',
    'haarcascade_profileface.xml',
]

face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + classifiers_list[3])


img_data_list = []
total_number_faces = 0
for img_path in img_path_list:
    name, ext = os.path.splitext(img_path)
    if ext in ['.jpg', '.jpeg', '.png'] and name != 'x':
        number_of_faces = name.split('-')
        
        img_data_list.append({
            "path": os.path.join(IMGS_DIR, img_path),
            "number_of_faces": int(number_of_faces[0]),
            "number_of_detected_faces": 0,
            "Accuracy": 0
        })
        
        total_number_faces = total_number_faces + int(number_of_faces[0])

total_number_detected = 0
for img_data in img_data_list:
    img = cv2.imread(img_data["path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (750,750))
    
    faces = face_detection.detectMultiScale(img,1.05,1)  # 1.05, 55 -> No Resize  # 
    # img , scale_factor, nearest neighbor
    
    img_data["number_of_detected_faces"] = len(faces)
    total_number_detected = total_number_detected + len(faces)
    img_data["Accuracy"] = len(faces)/img_data["number_of_faces"] * 100
    

end = time()

img_data_list.append({"total_time": end - start})
img_data_list.append({
    "avg_accuracy": total_number_detected/total_number_faces*100,
    'total_number_faces' : total_number_faces,
    'total_number_detected': total_number_detected
    })

with open('Haar-Cascades/classifiers/results/profile_resize.json', "w") as outfile:  
    json.dump(img_data_list, outfile)
    
print("Done")
