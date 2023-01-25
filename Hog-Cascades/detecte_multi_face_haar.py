# Import packages
import cv2
import matplotlib.pyplot as plt
import os

FOLDER_NAME = "test_images"
IMG_NAME = '2.jpg'
CLASSIFIER_BASE_DIR = 'Haar-Cascades-Cuda\\haarcascades_cuda\\'

face_detection = cv2.CascadeClassifier(CLASSIFIER_BASE_DIR + 'haarcascade_frontalface_default.xml')

img_path = os.path.join(FOLDER_NAME, IMG_NAME)
img = cv2.imread(img_path)
img = cv2.resize(img, (750, 750))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detection.detectMultiScale(img_gray,1.009,20)  # 1.05, 55 -> No Resize  # 
# img , scale_factor, nearest neighbor

n = len(faces)
fig, ax = plt.subplots(ncols=n+2)  # ax is a list of axes
fig.suptitle(f'Multi Face Detection ({n})')

ax[0].imshow(img)
ax[0].set_title("Original")

next = 2 
for x,y, w, h in faces: 
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)  # Overwrite the original image in memory
    ax[1].imshow(img)
    ax[1].set_title("Detected")
    
    # Save img
    name, ext = os.path.splitext(IMG_NAME)
    detected_img_path = os.path.join(FOLDER_NAME, name+"-detected"+ext)
    cv2.imwrite(detected_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # cv2 expects BGR img so must convert back
    
    cropped_img = img[y:y+h, x:x+w] # Y -> X
    ax[next].imshow(cropped_img)
    ax[next].set_title(f"Person {next - 1}")
    next += 1
    
plt.show()
    
    



