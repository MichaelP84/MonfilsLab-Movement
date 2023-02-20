import cv2
import numpy as np


video = './SUNP0007/SUNP0007.MOV'
cap = cv2.VideoCapture(video)

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

ret, frame = cap.read()

if ret == True:
    cv2.imshow('Frame',frame)

    # Filename
    filename = 'savedImage.jpg'
      
    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename, frame)

    if cv2.waitKey(2000) & 0xFF == ord('q'):
        cap.release()
 
# Closes all the frames
cv2.destroyAllWindows() 


