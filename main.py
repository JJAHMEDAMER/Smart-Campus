import cv2

# Video Stream
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    cv2.imshow('frame_vscode', frame)
    
    if cv2.waitKey(1) == ord("q"): # press q to close
        break
    
cam.release
cv2.destroyAllWindows()
