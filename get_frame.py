import cv2
import os
import numpy as np
import specifications

frames_to_skip = specifications.minutes_to_skip * 60 * 60
saved_path = 'Screenshots'
full_dir = os.getcwd()
directory = os.path.join(full_dir, saved_path)

if (not os.path.exists(saved_path)):
  os.mkdir(saved_path)

for i, video in enumerate(os.listdir(specifications.video_path)):
  f = os.path.join(specifications.video_path, video)
  print(f)
  
  cap = cv2.VideoCapture(f)

  if (cap.isOpened() == False): 
    print("Error opening video stream or file")

  cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)
  ret, frame = cap.read()

  if ret == True:

      # Filename
      filename = (f'savedImage_{i}.jpg')
      
      # Using cv2.imwrite() method
      # Saving the image
      cv2.imwrite(os.path.join(directory, filename), frame)


# Closes all the frames
cv2.destroyAllWindows() 
