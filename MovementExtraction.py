import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import math
import shutil
import cv2
# from matplotlib import pyplot as plt
from multiprocessing import Pool, Process
import itertools
# import time
import os
import specifications

# general
minutes_to_skip = specifications.minutes_to_skip
frames_per_sec = specifications.frames_per_sec
frames_to_skip = minutes_to_skip * frames_per_sec * frames_per_sec
pixel_to_inch = specifications.pixel_to_inch
debug = specifications.debug
titles = specifications.titles
download_raw_approaches = specifications.download_raw_approaches

# Blue color in BGR
color = (255, 0, 0)

def main():
  
  # cluster behavior
  cluster_threshold = specifications.cluster_threshold
  
  # approach behavoir
  approach_distance = specifications.approach_distance
  approach_time = specifications.approach_time
  null_frame_tolerance = specifications.null_frame_tolerance
  
  # velocity and distance
  velocity_bin = specifications.velocity_bin
  
  # points
  # bodyDict = {0: 'nose', 2: 'right_ear', 4: 'left_ear', 6: 'back1', 8: 'back2', 10: 'back3', 12: 'back4', 14: 'tail_base', 16: 'tail_mid', 18: 'tail_end'}
    
  csv_path = specifications.csv_path
  video_type = specifications.video_type
  video_path = specifications.video_path

  
  if (debug):
    print("Running in verification mode...") 
  else:
    print("Not running in verification mode...")
    
  csv_list = []
  file_names = []
  
  # create a folder to house all the analysis
  directory = os.getcwd()
  folder = "Analysis"
  main_path = os.path.join(directory, folder)

  if not os.path.exists(main_path):
    os.mkdir(main_path)
  
  # create a list of videos for cross checking analysis in debug mode
  video_list = []
  if (debug):
    for filename in os.listdir(video_path):
      f = os.path.join(video_path, filename)
      if os.path.isfile(f):
        video_list.append(f)

  # adds all the csv paths to a list
  for filename in os.listdir(csv_path):      
    file_names.append(filename)
    f = os.path.join(csv_path, filename)
    if os.path.isfile(f):
      csv_list.append(f)
      
  for i, csv_path in enumerate(csv_list):
    print("\n\n\n----------File {}/{}-----------".format(i + 1, len(csv_list)))

    # create a file for every csv
    filename = file_names[i][: -4]
    working_directory = os.path.join(main_path, filename)
    if (debug):
      video_path = video_list[i]
    
    if not os.path.exists(working_directory):
      os.mkdir(working_directory)
    
    # read in csv file
    data = pd.read_csv(csv_path)
    data = data.iloc[:, 1:]

    # get lists of unique individuals and bodyparts
    individuals = np.unique(data.iloc[0].values)
    bodyparts = np.unique(data.iloc[1].values)
    num_individuals = individuals.size
    num_bodyparts = bodyparts.size
    
    individual_pd = []

    #1. creating a list of data sets where each index is a different animal
    for i, individual in enumerate(individuals):
        individual_data = data.iloc[:, 3 * num_bodyparts * i:3 * num_bodyparts * (i + 1)]
        individual_pd.append(individual_data)
  
    #2. renaming column titles
    bodypart = individual_pd[0].iloc[1].values
    value = individual_pd[0].iloc[2].values
    column_titles = []
    for i in range (len(bodypart)):
        column_titles.append(bodypart[i] + '_' + value[i])
    for i in range(len(individual_pd)):
      individual_pd[i] = individual_pd[i].iloc[3:]
      individual_pd[i].columns = column_titles
    
    #3. dropping likelyhood columns
    for i in range (len(individual_pd)):
      columns_to_drop = []
      for x, column in enumerate(individual_pd[i].columns):
          if (x + 1) % 3 == 0:
              columns_to_drop.append(column)
      individual_pd[i].drop(columns_to_drop, inplace=True, axis=1)
    
    #3. skipping forward x minutes
    for i in range(len(individual_pd)):
        individual_pd[i] = individual_pd[i].iloc[frames_to_skip:]
        individual_pd[i].reset_index(drop=True, inplace=True)

    animals = []
    for i in range (len(individual_pd)):
      animals.append('rat_{}'.format(i))


    #4. calculating velocity and distance in bins
    # assuming the nose is the basis of the movement
    # curently skips over nan values
    velocity_path = os.path.join(working_directory, 'velocity.csv')
    distance_path = os.path.join(working_directory, 'total_distance.csv')
    if not os.path.exists(velocity_path) or not os.path.exists(distance_path):
      print("Calculating velocity...")
      work = []
      
      for i in range (len(individual_pd)):
        w = (i, individual_pd, velocity_bin)
        work.append(w)

      p = Pool(num_individuals)
      results = p.starmap(getVelocity, work)  
      
      velocity = pd.DataFrame()
      total_distance = pd.DataFrame()

      for i in range(len(results)):
        total_distance['rat_{}'.format(i)] = [results[i][1]]
        velocity['rat_{}'.format(i)] = results[i][0]
      
      velocity.to_csv(velocity_path)
      total_distance.to_csv(distance_path)

      print("\t ...done")
    
      if (debug):
        print(total_distance.head())
        print(velocity.head())
        
    else:
      print("Skipping Veloctity and Distance calculations because directory 'velocity.csv' and 'total_distance.csv' already exists")


    #5. Calculating Approach behaviour
    print("Calculating approach behavior...")
    directory = "Approach_Behavior"
    approach_directory = os.path.join(working_directory, directory)
    matrix_path = working_directory + '\\approach_matrix.csv'
    
    if not os.path.exists(matrix_path):
      # if download_raw_approaches and not os.path.exists(approach_directory):
      #   os.mkdir(approach_directory)
      #   print("Directory '{}' created at '{}' ".format(directory, approach_directory))

      work = []
      # num_individuals
      for i in range(num_individuals):
        
        rat_path = None
        if (download_raw_approaches):
          rat_path = approach_directory
          directory = "rat_{}".format(i)
          rat_path = os.path.join(rat_path, directory)
          
          if not os.path.exists(rat_path):
            os.mkdir(rat_path)

        x = (approach_distance, approach_time, null_frame_tolerance, i, individual_pd, rat_path)
        work.append(x)
      
      p = Pool(num_individuals)
      saved = p.starmap(trackRatX, work)
      
      # debugging verify frames
      saved_frames = []
      debugging_approaches = []
      approach_matrix = np.array([], dtype=np.intc)
      for x in saved:
        saved_frames.append(x[0])
        debugging_approaches.append(x[1])
        approach_matrix = np.append([approach_matrix], x[2])
      
      print (saved_frames)

      approach_matrix = approach_matrix.reshape(num_individuals, num_individuals)
      if (debug):
        print(f'approach matrix: {approach_matrix}')
      matrix_df = pd.DataFrame(approach_matrix, columns=animals,)
      matrix_df = matrix_df.set_axis(animals, axis ='index')
      matrix_df.to_csv(matrix_path)
               
      print("\t ...done")
      if (debug):
        print((saved_frames[0][0]))
        print((debugging_approaches[0][0]))
      
      if (debug):
        cap = cv2.VideoCapture(video_path)
        for (group_of_frames, group_of_approach) in zip(saved_frames, debugging_approaches):
          for (f, approach) in zip (group_of_frames, group_of_approach):
            f = int(f)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)            
            
            if (cap.isOpened() == False): 
              print("Error opening video stream or file")
            ret, frame = cap.read()

            center_coordinates = (int(approach[0]), int(approach[1]))
            radius = int(convertInchToPixel(approach_distance))
            frame = cv2.circle(frame, center_coordinates, radius, color, 2)

            if ret == True:
              # Display the resulting frame
              print("Frame: {}, Nose: {}".format(f, approach))
              cv2.imshow('Frame', frame)
              
              # press any button to go to next frame
              cv2.waitKey(0)
              
        cap.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()
    
    else:
      print("Skipping Approach Behavior calculations because directory 'Approach_Behavior' already exists")
    
    
    
    #6. creating an average position point, not including tail data
    # each animal's calcualtions is done simultanesouly on a different thread
    directory = "Average_Position"  
    path = os.path.join(working_directory, directory)
        
    # if the avg position calculations have not been run already, run them
    if not os.path.exists(path):
      #shutil.rmtree(path) deletes a folder
      os.mkdir(path)
      print("Directory '{}' created at '{}' ".format(directory, path))
      work = []
      print(video_path)
      for i in range (len(individual_pd)):
        x = (individual_pd, i)
        work.append(x)
      
      p = Pool(num_individuals)
      average_pd = p.starmap(getAverage, work)  
      
      for i, avg in enumerate(average_pd):
        filename = "rat{}_avg.csv".format(i)
        single_path = os.path.join(path, filename)
        avg.to_csv(path_or_buf=single_path)

    
    # otherwise, load data from existing folder
    else:
      print("Skipping Average Point calculations because directory Average_Point exists")
      average_pd = []
      for i, csv in enumerate(os.listdir(path)):
        f = os.path.join(path, csv)
        data = pd.read_csv(f)
        data = data.drop(data.columns[0], axis=1)
        average_pd.append(data)
        
        
    #7. find clusters
    directory = "raw_cluster.csv"
    raw_path = os.path.join(working_directory, directory)
    
    directory = "cluster_totals.csv"
    cluster_path = os.path.join(working_directory, directory)
    
    if not os.path.exists(cluster_path) or not os.path.exists(raw_path):
          
      print("Clustering: ")
      all_clusters = []
      cluster_frames = []
      centriods = []
      all_points = []

      num_frames = len(average_pd[0])
      counts = [0] * (num_individuals)
      memory = []

      max_elligble = 0
      max_group = 0

      for frame in range (num_frames):
        if ((frame + 1) % 20 == 0):
          print(f'{frame + 1}/{num_frames}')
        # print(frame)
        Rat_Point_list = []
        
        # create a list of rat_point objects for each rat that is not null
        i = 0
        while i < len(individual_pd):
          avg_x1, avg_y1 = average_pd[i].iloc[frame, : ]
          if (not math.isnan(avg_x1)):
            rat = Rat_Point(avg_x1, avg_y1, str(i))
            Rat_Point_list.append(rat)
          i += 1

        max_elligble = max(len(Rat_Point_list), max_elligble)  

        # for group size 3 - 15 inclusize,
        # threshold dictionary defines given group sizes (key) and: their maximum centriod length (value)
        thresholds = {3:2.5, 4:4, 5:6, 6:6.5, 7:7.5, 8:8.5, 9:9, 10:9.5, 11:9.5, 12:9.75, 13:10, 14:10, 15:10}
        if (len(Rat_Point_list) >= 3):
          clusters = []
          # start at the max size group to skip over checking subgroups of large clusters that are identified
          for group_size in reversed(range(3, len(Rat_Point_list) + 1)):
            combinations = getCombinations(Rat_Point_list, group_size)
            
            for comb in combinations: 
              if (not contain(clusters, comb)):
                # if the combination is not already a subgroup of a cluster
                is_cluster, center_point, all_coords = isACluster(comb, thresholds[group_size], frame)
                if (is_cluster):
                  center_point.append(frame + frames_to_skip)
                  # if we havent seen this combination within the past frames (is not in memory)
                  # add it to memory
                  inMemory = False
                  for mem_comb in memory:
                    if (mem_comb.wraps(comb)):
                      mem_comb.subtract()
                      inMemory = True
                      
                  if (not inMemory):
                    c = Combination(cluster_threshold, comb, center_point, all_coords)
                    memory.append(c)
                            
                  clusters.append(comb)
                              
        # after looking at all cominations in this frame
        # see if a combination in memory dropped out
        for mem_comb in memory:
          if not mem_comb.isWithin(clusters) and not mem_comb.hasBuffer():
            if (mem_comb.isCleared()):
              # combination existance time satisfied, add it to found clusters
              all_clusters.append(mem_comb.getList())
              cluster_frames.append(mem_comb.getFrames())              
              centriods.append(mem_comb.getCentriod())
              all_points.append(mem_comb.getAllCoords())

              # max_group = max(max_group, len(mem_comb.getList()))
              
              # list of total group sizes
              counts[len(mem_comb.getList()) - 1] += 1
            memory.remove(mem_comb)
            
          elif not mem_comb.isWithin(clusters) and mem_comb.hasBuffer():
            # combinaiton in memory is not found in this frame:
            # 4 buffer frames allow a cluster to have some missing point data 
            mem_comb.buffer()
            mem_comb.addFrame(frame + frames_to_skip)
            
          else:
            # combination in memory found again
            mem_comb.addFrame(frame + frames_to_skip)   
              
      print(f'A max number of {max_elligble} rats were seen on screen at one time')
      print(f'A max group of size {max_group} was found on screen at one time')


      if (debug):
        proportion_in_cluster = len(cluster_frames) / num_frames
        print("{} of all frames have a cluster".format(proportion_in_cluster))
        print('length of all clusters: ', len(all_clusters))
        print('lengths of all cluster_frames: ', len(cluster_frames))
      
      # create csv of all clusters
      cluster_csv = pd.DataFrame()
      cluster_csv['clusters'] = all_clusters
      initial_time = []
      durations = []
      for group in cluster_frames:
        initial_time.append(getTime(group[0])) # as min, sec
        durations.append(len(group)/60) # as seconds
        
      cluster_csv['time_stamp'] = initial_time
      cluster_csv['duration_seconds'] = durations
      cluster_csv.to_csv(raw_path)
      
      # create a csv of just counts of cluster amounts                   
      cluster_totals = pd.DataFrame()
      for i in range(num_individuals):
        cluster_totals[f'size_{i + 1}'] = [counts[i]]
      cluster_totals.to_csv(cluster_path)
      
      if (debug):

        # testing code: manually check frames at identified clusters
        for (group, clusters, centers, points) in zip(cluster_frames, all_clusters, centriods, all_points):
          x_pos, y_pos, length = centers[0], centers[1], centers[2]
          print(x_pos, y_pos, length)

          for f in group:
            print(f)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)

            if (not cap.isOpened()): 
              print("Error opening video stream or file")
            
            ret, img = cap.read()
            if ret == True:
              radius = int(convertInchToPixel(length))
              img = cv2.circle(img, (int(x_pos), int(y_pos)), radius, color, 2)
              for point in points:
                print(int(point[0]), int(point[1])) # getting points at the exact same spot????
                img = cv2.circle(img, (int(point[0]), int(point[1])), 10, color, 4)
              
              # Display the resulting frame
              printTime(f)
              print("Group size: {}".format(len(clusters)))
              cv2.imshow('Frame', img)
              # press any button to go to next frame
              cv2.waitKey(0)
            
            cap.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()
    
    else:
      print("Skipping Cluster calculations because directory Raw_Clusters exists")

    
class Combination:
  def __init__(self, threshold: int, comb: list, centriod: tuple, all_coords:list) -> None:
    self.all_coords = all_coords
    self.centriod = centriod
    self.framesLeft = threshold
    self.bufferFrames = 4
    self.comb = comb
    self.frames = []
  
  def isWithin(self, newCombinations: list) -> bool:
    return any(group == self.comb for group in newCombinations)
  
  def subtract(self) -> None:
    if (self.framesLeft > 0):
      self.framesLeft -= 1
  
  def buffer(self):
    self.bufferFrames -= 1
  
  def hasBuffer(self) -> bool:
    return self.bufferFrames > 0
    
  def isCleared(self) -> bool:
    return self.framesLeft == 0
  
  def addFrame(self, f) -> None:
    self.frames.append(f)
    
  def getFrames(self) -> list:
    return self.frames
  
  def getCentriod(self) -> tuple:
    return self.centriod

  def wraps(self, comb: list) -> bool:
    return self.comb == comb
  
  def getList(self) -> list:
    return self.comb
  
  def getAllCoords(self) -> list:
    return self.all_coords

# returns the binned velocity for rats over time
def getVelocity(i: int, individual_pd: list, velocity_bin: int):  
  
  max_inputs = (len(individual_pd[i]) // velocity_bin)

  velocity_singular = [] #velocity list for a single animal
  total_distance = 0
  
  frame = 0
  while frame < len(individual_pd[i]):
      # get current x and y position
      prev_x, prev_y = individual_pd[i].iloc[frame, :2] # 2 refers to nose since they are the first 2 points
      # skipping over nan frames
      while (math.isnan(prev_x)):
              frame += 1
              prev_x, prev_y = individual_pd[i].iloc[frame, :2]
      
      # get next non-nan x and y position
      frame += velocity_bin
      if (frame < len(individual_pd[i])):
          next_x, next_y = individual_pd[i].iloc[frame, :2]
          
          frames_added = 0
          while (math.isnan(next_x) and frame < len(individual_pd[0])): # continue adding frames until next non-nan frame
              frame += 1
              frames_added += 1
              if (frame < len(individual_pd[0])):
                next_x, next_y = individual_pd[i].iloc[frame, :2]
          
          #calculate the distance traveled in the bin for total distance and velocity
          dist = getDistance(prev_x, prev_y, next_x, next_y)
          total_distance += dist
          
          time = (velocity_bin + frames_added) / frames_per_sec
          temp_velocity = dist / time
          velocity_singular.append(temp_velocity)
  
  while (len(velocity_singular) < max_inputs):
    velocity_singular.append(None)
    
  return velocity_singular, total_distance
  
      
# given a list of cluster and a new combination, return true if the combination
# is already within a cluster
# massive clusters take priority
def contain(clusters: list, comb: list) -> bool:
    for group in clusters:
      if (any(item in group for item in comb) == True):
        return True
    return False

# represents a rat's average point
class Rat_Point:

  def __init__(self, x: float, y: float, index: str):
    self.x = x
    self.y = y
    self.name = "rat_" + index
  
  def __eq__(self, other):
    if isinstance(other, self.__class__):
        return self.name == other.name
    else:
        return False
    
  def getCoordinates(self):
    return self.x, self.y

  def __str__(self):
    return self.name
  
  def __repr__(self):
    return self.name

  def getName(self):
    return self.name
  
  def isNan(self):
    return math.isnan(self.x)
  
  def getPointDistance(self, Rat_Point):
    other_x, other_y = Rat_Point.getCoordinates()
    return getDistance(self.x, self.y, other_x, other_y)
  

def getCombinations(arr: list, choose: int):
  combinations = []
  for comb in itertools.combinations(arr, choose):
    combinations.append(comb)
  
  return combinations
  

def isACluster(arr: list, threshold: int, frame: int) -> bool:
  coords = []
  sum_x = 0
  sum_y = 0
  for rat in arr:

    x, y = rat.getCoordinates()
    coords.append([x, y])
    sum_x += x
    sum_y += y
  
  avg_x = sum_x / len(arr)
  avg_y = sum_y / len(arr)
  

  # print(coords)

  # if (debug):
  #   vid = 'C:\\Users\\mrp3844\\MonfilsLab\\Analysis\\MonfilsLab-Movement\\Videos\\SUNP0009-2.MOV'
  #   cap = cv2.VideoCapture(vid)
  #   cap.set(cv2.CAP_PROP_POS_FRAMES, frame + frames_to_skip)

  #   ret, img = cap.read()
        
  #   if ret == True:

  #     for rat in coords:
  #       img = cv2.circle(img, (int(rat[0]), int(rat[1])), 50, color, 2)
  #     # Display the resulting frame
  #     cv2.imshow('Frame', img)
  #     print('Frame: ', frame)
      
  #     # press any button to go to next frame
  #     cv2.waitKey(0)
          
  #   cap.release()
  
  
  for rat in arr:
    rat_x, rat_y = rat.getCoordinates()
    dist = getDistance(rat_x, rat_y, avg_x, avg_y)
    if dist > threshold:
      # print("not a cluster, distance: {} > threshold {}".format(dist, threshold))
      return False, None, None
  return True, [avg_x, avg_y, threshold], coords
    

# for a given rat i, tracks its approach behavior on alll other rats
def trackRatX(approach_distance: float, approach_time: float, null_frame_tolerance: int, i: int, individual_pd: list, rat_path: str):
  # print('I am number %d in process %d' % (i, os.getpid()))
  print ("started task {}".format(i))

  # how many frames the rat should be < 'approach_distance' from another to count as an approach behaviours
  approach_frames = round(approach_time * frames_per_sec)  
  
  # each element in approach_singular is a DataFrame of interactions with the other rats
  approach_singular = []
  
  counts = np.zeros(len(individual_pd), dtype=np.intc)  
  saved_frames = []
  debugging_approaches = []
  
  for j in range (len(individual_pd)):
      # there is a column for every body parts, the rows indicate interactions
      # also a column for time of interactions
      approach_to_j = pd.DataFrame(columns=titles)
      
      #comparing every animal to each other, so exclude comparing to self
      path = rat_path
      if (i != j):
        frame = 0 # total frames = frame + skipped frames
        engagement_frames = 0 # number of frames less than a distance
        null_frames = 0
        # go through the whole video comparing rat i with rat j
        while frame < (len(individual_pd[i])):
            nose_x, nose_y = individual_pd[i].iloc[frame, :2]
            while (math.isnan(nose_x) and frame < len(individual_pd[i])):
                frame += 1
                if frame < len(individual_pd[i]):
                  nose_x, nose_y = individual_pd[i].iloc[frame, :2]

            if (frame < len(individual_pd[i])):
              
              bodyparts = individual_pd[j].iloc[frame, :-4] #exclude last four columns because they refer to points on the rat's tail
              
              min_index, min_distance, min_list = findMinDistance(bodyparts, nose_x, nose_y)
              # min_distance in inches

              if (not min_index is None and min_distance < approach_distance):
                engagement_frames += 1
                if (engagement_frames >= approach_frames):
                  
                  # for debugging keep a list of interactions and the frames they happen
                  if (debug):
                    ### print(f'Frame {frame + frames_to_skip} \t Engangement_frames {engagement_frames} > {approach_frames} \t {min_distance} < {approach_distance}')
                    adjusted_frame = frame + frames_to_skip
                    saved_frames.append(adjusted_frame)
                    debugging_approaches.append([nose_x, nose_y])
                  
                  counts[j] += 1
                  
                  if (download_raw_approaches):
                    min = pd.DataFrame(min_list).T
                    min.columns = titles
                    approach_to_j = pd.concat([approach_to_j, min], axis=0, ignore_index=True)
                  
                  engagement_frames = 0
                  null_frames = 0
                
              elif (min_index is None and null_frames < null_frame_tolerance):
                null_frames += 1

              else:
                engagement_frames = 0
                null_frames = 0
              
              frame += 1
            
        if (download_raw_approaches):
          approach_singular.append(approach_to_j)
        
      elif (download_raw_approaches):
          approach_singular.append(['same animal'])

      if (download_raw_approaches):
        filename = "rat{}_to_rat{}.csv".format(i, j)
        path = os.path.join(path, filename)
        approach_to_j.to_csv(path_or_buf=path)
  
  print ("finished task {}".format(i))
  return saved_frames, debugging_approaches, counts
  
# from a list of x and y points for body parts, return the min, its distance (inches) and its index
def findMinDistance(bodyparts: list, nose_x: float, nose_y: float):
    min_index = -1
    min_distance = 1_000
    min_list = [None] * (round(len(bodyparts) / 2))
    # divide by 2 because we are going from a list with two columns for every body point (x and y) to just every body point

    i = 0
    while (i < len(bodyparts)):
        part_x = bodyparts[i]
        part_y = bodyparts[i + 1]
        
        if (not part_x is None):
            distance = getDistance(nose_x, nose_y, part_x, part_y)
            if (distance < min_distance):
                min_distance = distance
                min_index = i
            
        i += 2
    
    if (min_index == -1):
        return None, None, min_list
    
    min_list[round(min_index/2)] = min_distance
    return min_index, min_distance, min_list
  
      
# returns the distance between two points (inches)
def getDistance(x1, y1, x2, y2):
    return convertPixelToInch(math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2)))

# converts a pixel length to inches
def convertPixelToInch(x):
  return x / pixel_to_inch

# converts a inch to pixel
def convertInchToPixel(x):
  return x * pixel_to_inch

# returns the average x and average y point of a list of points (x1, y1, x2, y2, x3, ...)    
def getAverage(individual_pd, i):
  average_x = []
  average_y = []

  print ("started task {}".format(i))
  for frame in range (len(individual_pd[i])):
    sum_x = 0
    sum_y = 0
    count = 0
    
    for j in range (len(individual_pd[i].columns) - 4): # minus four columns to remove tail points
      if j % 2 == 0:    
        pos_x = individual_pd[i].iloc[frame, j]

        if (not math.isnan(pos_x)):
          sum_x += pos_x
          count += 1
      else:
        pos_y = individual_pd[i].iloc[frame, j]
          
        if (not math.isnan(pos_y)):
          sum_y += pos_y
      
    if (count == 0):
      average_x.append(None)
      average_y.append(None)
    else:
      avg_x = sum_x / count 
      average_x.append(avg_x)
      avg_y = sum_y / count
      average_y.append(avg_y)

    
  average_pd = pd.DataFrame()
  average_pd['average_x'] = average_x
  average_pd['average_y'] = average_y
  print ("finished task {}".format(i))
  
  return average_pd

def time_convert(sec) -> None:
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))
  
def printTime(frame: int) -> None:
  sec = frame // 60 % 60
  min = frame // 3600
  print(f'Frame: {frame}, Minute: {min}, Second: {sec}')
  
def getTime(frame: int) -> None:
  sec = frame // 60 % 60
  min = frame // 3600
  return(f'{min}:{sec}')

if __name__ == "__main__":
 main()
