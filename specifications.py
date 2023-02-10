# This file houses all the specifications related to analysis for the program
# make sure this file is in the same folder as MovementExtraction.py

csv_path = 'C:\\Users\\micha\\MonfilsLab\\analysis\\csv' # has to be double backslahes
video_path = 'C:\\Users\\micha\\MonfilsLab\\analysis\\Videos' # has to be double backslahes
video_type = '.mp4' # '.avi' '.mp4' ... etc.
debug = False # (True/False)

pixel_to_inch = 22.75
frames_per_sec = 60
minutes_to_skip = 3

# Points (same order as you outlined in the config.yaml file of the deeplabcut project)
titles = ['nose', 'right_ear', 'left_ear', 'back1', 'back2', 'back3', 'back4', 'tail_base', 'tail_mid', 'tail_end']

# Velocity and distance
velocity_bin = 600 # (frames) 

# Clustering
cluster_threshold = 15 # (frames)

# Approach Behavior
approach_distance = 0.5 # (inches) 
approach_time = 0.5 # (seconds)
null_frame_tolerance = 20 # (frames) 