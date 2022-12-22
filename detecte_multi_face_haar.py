# Import packages
import cv2
import matplotlib.pyplot as plt

face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('test_multi_face2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detection.detectMultiScale(img_gray,1.9,5)
# img , scale_factor, nearest neighbor

n = len(faces)
fig, ax = plt.subplots(ncols=n+2)  # ax is a list of axes
fig.suptitle(f'Multi Face Detection ({n})')

ax[0].imshow(img)
ax[0].set_title("Original")

next = 2 
for x,y, w, h in faces: 
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 20)  # Overwrite the original image in memory
    ax[1].imshow(img)
    ax[1].set_title("Detected")
    cv2.imwrite("test.jpg", img)
    
    cropped_img = img[y:y+h, x:x+w] # Y -> X
    ax[next].imshow(cropped_img)
    ax[next].set_title(f"Person {next - 1}")
    next += 1
    
plt.show()
    
    



