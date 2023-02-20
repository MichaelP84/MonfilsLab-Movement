import cv2
import os
import numpy as np


saved_path = 'Screenshots'
full_dir = os.getcwd()
directory = os.path.join(full_dir, saved_path)


if (not os.path.exists(saved_path)):
  os.mkdir(saved_path)


for video in os.listdir('./Videos'):
  f = os.path.join(directory, video)

  print(f)

# video = './SUNP0007/SUNP0007.MOV'
# cap = cv2.VideoCapture(video)

# if (cap.isOpened()== False): 
#   print("Error opening video stream or file")

# ret, frame = cap.read()

# if ret == True:
#     cv2.imshow('Frame',frame)

#     # Filename
#     filename = 'savedImage.jpg'
      
#     # Using cv2.imwrite() method
#     # Saving the image
#     cv2.imwrite(filename, frame)

#     if cv2.waitKey(2000) & 0xFF == ord('q'):
#         cap.release()
 
# # Closes all the frames
# cv2.destroyAllWindows() 
