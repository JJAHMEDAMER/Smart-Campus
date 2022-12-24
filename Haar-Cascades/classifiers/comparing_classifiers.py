import cv2
from time import time
import matplotlib.pyplot as plt

print("running...")

start = time()

IMG_PATH = 'test_images/13.jpg'

img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

classifiers_list = [
    'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_alt_tree.xml',
    'haarcascade_frontalface_default.xml',
    'haarcascade_profileface.xml',
]


fig, axs = plt.subplots(ncols=3, nrows=2)  # ax is a list of axes
fig.suptitle(f'Haar Cascade classifier')

flatten_axs = axs.ravel()

flatten_axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
flatten_axs[0].set_title("Original", fontsize=10)

index = 0
for classifier in classifiers_list:
    
    face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + classifier)
    faces = face_detection.detectMultiScale(img,1.01,1)
    
    for x, y, w, h in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 10)
    
    n = len(faces)
    index = index + 1
    flatten_axs[index].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    flatten_axs[index].set_title(classifier + f'Detected Faces: {n}', fontsize=10)
    
    
     
    
plt.show()
    
    

    

