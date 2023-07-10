# This file houses all the specifications related to analysis for the program
# make sure this file is in the same folder as MovementExtraction.py

# has to be double backslahes
csv_path = 'C:\\Users\\mrp3844\\MonfilsLab\\Analysis\\MonfilsLab-Movement\\csv'
# has to be double backslahes
video_path = 'C:\\Users\\mrp3844\\MonfilsLab\\Analysis\\MonfilsLab-Movement\\Videos'
analysis_path = 'C:\\Users\\mrp3844\\MonfilsLab\\Analysis\\MonfilsLab-Movement\\Analysis'
video_type = '.mp4'  # '.avi' '.mp4' ... etc.
debug = False  # (True/False)
# downloads images of clusters that fialed to be classified becasue they were just outside the radius threshold
cluster_debug = False
# used for tuning the radius's

pixel_to_inch = 17.9701  # pixel to inch conversion
# 808.6557 pixels length
# old: 22.75        {3:4, 4:4.5, 5:6, 6:6.5, 7:7.5, 8:8.5, 9:9, 10:9.5, 11:10, 12:10.5, 13:11, 14:11.5, 15:12}
# 45 inch: 17.9701  {3: 5.063967368016873, 4: 5.696963289018982, 5: 7.595951052025309, 6: 8.22894697302742, 7: 9.494938815031636, 8: 10.760930657035855, 9: 11.393926578037965, 10: 12.026922499040072, 11: 12.659918420042182, 12: 13.292914341044291, 13: 13.9259102620464, 14: 14.55890618304851, 15: 15.191902104050618}
thresholds = {3: 5.063967368016873, 4: 5.696963289018982, 5: 7.595951052025309, 6: 8.22894697302742, 7: 9.494938815031636, 8: 10.760930657035855,
              9: 11.393926578037965, 10: 12.026922499040072, 11: 12.659918420042182, 12: 13.292914341044291, 13: 13.9259102620464, 14: 14.55890618304851, 15: 15.191902104050618}

frames_per_sec = 60  # fps of input video
minutes_to_skip = 3  # minutes of video to skip analyzing

# Points (same order as you outlined in the config.yaml file of the deeplabcut project)
titles = ['nose', 'right_ear', 'left_ear', 'back1', 'back2',
          'back3', 'back4', 'tail_base', 'tail_mid', 'tail_end']

# Velocity and distance
# (frames) determines the bins for veloctiy and distance calculation
velocity_bin = 600
avg_point_only = True  # runs only average point part of program

# Clustering
# (frames) number of frames to determine if a cluster exists
cluster_threshold = 15

# Approach Behavior
# (inches) distance to determine the approach behavior of animals
approach_distance = 1.5
# (seconds) time of interation to detemine if an approach behavior occurred
approach_time = 0.25
# (frames) number of missing point frames to allow in determining an approahc behavior
null_frame_tolerance = 5
# download csv of raw approached (warning: very large)
download_raw_approaches = False
