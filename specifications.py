# This file houses all the specifications related to analysis for the program
# make sure this file is in the same folder as MovementExtraction.py

csv_path = 'C:\\Users\\micha\\MonfilsLab\\analysis\\csv' # has to be double backslahes
video_path = 'C:\\Users\\micha\\MonfilsLab\\analysis\\Videos' # has to be double backslahes
video_type = '.mp4' # '.avi' '.mp4' ... etc.
debug = True # (True/False)

pixel_to_inch = 22.75 # pixel to inch conversion
frames_per_sec = 60 # fps of input video
minutes_to_skip = 3 # minutes of video to skip analyzing

# Points (same order as you outlined in the config.yaml file of the deeplabcut project)
titles = ['nose', 'right_ear', 'left_ear', 'back1', 'back2', 'back3', 'back4', 'tail_base', 'tail_mid', 'tail_end']

# Velocity and distance
velocity_bin = 600 # (frames) determines the bins for veloctiy and distance calculation

# Clustering
cluster_threshold = 15 # (frames) number of missing point frames to allow in determining if a cluster exists

# Approach Behavior
approach_distance = 1.5 # (inches) distance to determine the approach behavior of animals
approach_time = 0.25 # (seconds) time of interation to detemine if an approach behavior occurred
null_frame_tolerance = 5 # (frames) number of missing point frames to allow in determining an approahc behavior
download_raw_approaches = False # download csv of raw approached (warning: very large)