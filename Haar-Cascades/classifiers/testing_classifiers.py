import json
import cv2
import os

IMGS_DIR = 'test_images'
img_path_list = os.listdir(IMGS_DIR)

face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


img_data_list = []
for img_path in img_path_list:
    name, ext = os.path.splitext(img_path)
    number_of_faces = name.split('-')
    
    img_data_list.append({
        "path": os.path.join(IMGS_DIR, img_path),
        "number_of_faces": int(number_of_faces[0]),
        "number_of_detected_faces": 0,
        "Accuracy": 0
    })

for img_data in img_data_list:
    img = cv2.imread(img_data["path"])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_detection.detectMultiScale(img_gray,1.05,1)  # 1.05, 55 -> No Resize  # 
    # img , scale_factor, nearest neighbor
    
    img_data["number_of_detected_faces"] = len(faces)
    img_data["Accuracy"] = len(faces)/img_data["number_of_faces"] * 100
    
with open('Haar-Cascades/classifiers/test_noResize.json', "w") as outfile:  
    json.dump(img_data_list, outfile)