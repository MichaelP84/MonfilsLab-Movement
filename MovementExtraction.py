import pandas as pd
import numpy as np
import math
import shutil
import cv2
# from matplotlib import pyplot as plt
from multiprocessing import Pool, Process
import itertools
# import time
import os
import specifications

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
  
  titles = specifications.titles
  titles.append('time')
    
  csv_path = specifications.csv_path
  pixel_to_inch = specifications.pixel_to_inch
  minutes_to_skip = specifications.minutes_to_skip
  frames_per_sec = specifications.frames_per_sec
  debug = specifications.debug
  
  frames_to_skip = minutes_to_skip * frames_per_sec * frames_per_sec

  
  if (debug):
    print("Running in verification mode...")
    csv_path = input("\nEnter the full file path of the folder containig the video csv's: ")
  else:
    debug = False
    print("Not running in verification mode...")
    
  csv_list = []
  file_names = []
  
  # create a folder to house all the analysis
  directory = os.getcwd()
  folder = "Analysis"
  main_path = os.path.join(directory, folder)

  if not os.path.exists(main_path):
    os.mkdir(main_path)
  
  if (debug):
    video_list = []
  
  # adds all the csv paths to a list
  for filename in os.listdir(csv_path):
    # if (verification_mode):
    #   video_list.append()
      
    file_names.append(filename)
    f = os.path.join(csv_path, filename)
    if os.path.isfile(f):
      csv_list.append(f)
      
  for i, csv_path in enumerate(csv_list):
    print("\n\n\n----------File {}/{}-----------".format(i + 1, len(csv_list)))

    # create a file for every csv
    filename = file_names[i][: -4]
    working_directory = os.path.join(main_path, filename)
    
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
  


    #4. calculating velocity and distance in bins
    # assuming the nose is the basis of the movement
    # curently skips over nan values
    velocity_path = os.path.join(working_directory, 'velocity.csv')
    distance_path = os.path.join(working_directory, 'total_distance.csv')
    if not os.path.exists(velocity_path) or not os.path.exists(distance_path):
      print("Calculating velocity...")
      work = []
      animals = []
      
      for i in range (len(individual_pd)):
        w = (i, individual_pd, velocity_bin, frames_per_sec, pixel_to_inch)
        work.append(w)
        animals.append('rat_{}'.format(i))
      
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
    
    if not os.path.exists(approach_directory):
      os.mkdir(approach_directory)
      print("Directory '{}' created at '{}' ".format(directory, approach_directory))

      work = []
      # num_individuals
      for i in range(num_individuals):
        rat_path = approach_directory
        directory = "rat_{}".format(i)
        rat_path = os.path.join(rat_path, directory)
        
        if not os.path.exists(rat_path):
          os.mkdir(rat_path)

        x = (approach_distance, approach_time, null_frame_tolerance, i, individual_pd, frames_to_skip, frames_per_sec, rat_path, pixel_to_inch, titles)
        work.append(x)
      
      p = Pool(num_individuals)
      saved = p.starmap(trackRatX, work)
      
      # debugging verify frames
      saved_frames = []
      debugging_approaches = []
      for x in saved:
        saved_frames.append(x[0])
        debugging_approaches.append(x[1])
      
      print("\t ...done")
    
    else:
      print("Skipping Approach Behavior calculations because directory 'Approach_Behavior' already exists")
    
    # video_path = 'C:\\Users\\micha\\MonfilsLab\\analysis\\Videos\\SUNP0007DLC_resnet50_MovementLabDec16shuffle1_350000_full.mp4'

    # for (group_of_frames, group_of_approach) in zip(saved_frames, debugging_approaches):
      
    #   for (f, approach) in zip (group_of_frames, group_of_approach):
        
    #     cap = cv2.VideoCapture(video_path)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        
    #     if (cap.isOpened()== False): 
    #       print("Error opening video stream or file")
    #     ret, frame = cap.read()
        
    #     if ret == True:
    #       # Display the resulting frame
    #       print("Frame: {}, {}".format(f, approach))
    #       cv2.imshow('Frame', frame)
    #       # press any button to go to next frame
    #       cv2.waitKey(0)
          
    # cap.release()
    
    # # Closes all the frames
    # cv2.destroyAllWindows()
      
    # print("\t ...done")
    
    #6. creating an average position point, not including tail data
    # doing so for each animal is done simultanesouly on a different thread

    #creating and start a list of tasks (getAverage)
    
    directory = "Average_Position"  
    path = os.path.join(working_directory, directory)
        
    # if the avg position calculations have not been run already, run them
    if not os.path.exists(path):
      #shutil.rmtree(path) deletes a folder
      os.mkdir(path)
      print("Directory '{}' created at '{}' ".format(directory, path))
      work = []

      for i in range (len(individual_pd)):
        x = (individual_pd, i)
        work.append(x)
      
      p = Pool(num_individuals)
      average_pd = p.starmap(getAverage, work)  
      
      for i, avg in enumerate(average_pd):
        filename = "rat{}_avg.csv".format(i)
        single_path = os.path.join(path, filename)
        avg.to_csv(path_or_buf=single_path)
        print(avg)
    
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
      num_frames = len(average_pd[0])
      counts = [0] * (num_individuals)
      memory = []
      
      for frame in range (num_frames):
        # print("Frame{}/{}".format(frame, len(individual_pd[0])))
        Rat_Point_list = []
        
        # create a list of rat_point objects for each rat that is not null
        i = 0
        while i < len(individual_pd):
          avg_x1, avg_y1 = average_pd[i].iloc[frame, : ]      
          if (not math.isnan(avg_x1)):
            rat = Rat_Point(avg_x1, avg_y1, str(i))
            Rat_Point_list.append(rat)
          i += 1
          
        # print("Number of rats: {}".format(len(Rat_Point_list)))   

        # for group size 3 - 9 inclusize,
        # threshold dictionary defines given group sizes (key) and: their maximum centriod length (value)
        thresholds = {3: 1, 4: 2, 5: 3, 6: 3.8, 7:6, 8:7, 9:8}
        if (len(Rat_Point_list) >= 3):
          clusters = []
          # start at the max size group to skip over checking subgroups of large clusters that are identified
          for group_size in reversed(range(3, len(Rat_Point_list) + 1)):
            combinations = getCombinations(Rat_Point_list, group_size)
            # print("For group size: {}".format(group_size))

            for comb in combinations: 
              # if the combination is not already a subgroup of a cluster
              if (not contain(clusters, comb)):
                if (isACluster(comb, thresholds[group_size], pixel_to_inch)):
                  # print("\tcluster size: {}, found".format(group_size))
                  # print("\t\tcontains the following:")
                  
                  # if we havent seen this combination within the past frames (is not in memory)
                  # add it to memory
                  inMemory = False
                  for mem_comb in memory:
                    if (mem_comb.wraps(comb)):
                      mem_comb.subtract()
                      inMemory = True
                  if (not inMemory):
                    c = Combination(cluster_threshold, comb)
                    memory.append(c)
                            
                  clusters.append(comb)
                  
                  # print("\t\t{}".format(rat.getName()))
            
        # print(f'memory: {memory}')

        # after looking at all cominations in this frame
        # see if a combination in memory dropped out
        for mem_comb in memory:
          if not mem_comb.isWithin(clusters) and not mem_comb.hasBuffer():
            if (mem_comb.isCleared()):
              all_clusters.append(mem_comb.getList())
              cluster_frames.append(mem_comb.getFrames())
              counts[len(mem_comb.getList()) - 1] += 1
              
            memory.remove(mem_comb)
          elif not mem_comb.isWithin(clusters) and mem_comb.hasBuffer():
            # 4 buffer frames allow a cluster to have some missing point data 
            mem_comb.buffer()
            mem_comb.addFrame(frame + frames_to_skip)
          else:
            mem_comb.addFrame(frame + frames_to_skip)
      
      proportion_in_cluster = len(cluster_frames) / num_frames
      print("{} of all frames have a cluster".format(proportion_in_cluster))
      print(len(all_clusters))
      print(len(cluster_frames))
      
      # create csv of all clusters
      cluster_csv = pd.DataFrame()
      cluster_csv['clusters'] = all_clusters
      cluster_csv['frame'] = cluster_frames
      cluster_csv.to_csv(raw_path)
      
      # create a csv of just counts of cluster amounts                   
      cluster_totals = pd.DataFrame()
      for i in range(num_individuals):
        print(counts[i])
        cluster_totals[f'size_{i + 1}'] = [counts[i]]
      cluster_totals.to_csv(cluster_path)
    
    else:
      print("Skipping Cluster calculations because directory Average_Point exists")

      
    # # testing code: manually check frames at identified clusters    
    # video_path = 'C:\\Users\\micha\\MonfilsLab\\analysis\\Videos\\SUNP0007DLC_resnet50_MovementLabDec16shuffle1_350000_full.mp4'

    # for (f, clusters) in zip(cluster_frames, all_clusters):
      
    #   cap = cv2.VideoCapture(video_path)
    #   cap.set(cv2.CAP_PROP_POS_FRAMES, f)
      
    #   if (cap.isOpened()== False): 
    #     print("Error opening video stream or file")
    #   ret, frame = cap.read()
      
    #   if ret == True:
    #     # Display the resulting frame
    #     printTime(f)
    #     print("Group size: {}".format(len(clusters[0])))
    #     cv2.imshow('Frame', frame)
    #     # press any button to go to next frame
    #     cv2.waitKey(0)
        
    # cap.release()
    
    # # Closes all the frames
    # cv2.destroyAllWindows()

class Combination:
  def __init__(self, threshold: int, comb: list) -> None:
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
  
  def wraps(self, comb: list) -> bool:
    return self.comb == comb
  
  def getList(self) -> list:
    return self.comb
  
def getVelocity(i, individual_pd: list, velocity_bin: int, frames_per_sec: float, pixel_to_inch: float):  
  
  max_inputs = (len(individual_pd[i]) // velocity_bin)

  velocity_singular = [] #velocity list for a single animal
  total_distance = 0
  
  frame = 0
  while frame <= len(individual_pd[i]):
      # get current x and y position
      prev_x, prev_y = individual_pd[i].iloc[frame, :2] # 2 refers to nose, -2 would be avg (columns in list)
      # skipping over nan frames
      while (math.isnan(prev_x)):
              frame += 1
              prev_x, prev_y = individual_pd[i].iloc[frame, :2]
      
      # get next non-nan x and y position
      frame += velocity_bin
      if (frame <= len(individual_pd[i])):
          next_x, next_y = individual_pd[i].iloc[frame, :2]
          
          frames_added = 0
          while (math.isnan(next_x)): # continue adding frames until next non-nan frame
              frame += 1
              frames_added += 1
              next_x, next_y = individual_pd[i].iloc[frame, :2]
          
          #calculate the distance traveled in the bin for total distance and velocity
          dist = getDistance(prev_x, prev_y, next_x, next_y, pixel_to_inch)
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
        # for x in group:
        #   print(x)
        # print("---Contains---")
        # for y in comb:
        #   print(y)
        # print("--------------")
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
  
  def getPointDistance(self, Rat_Point, pixel_to_inch):
    other_x, other_y = Rat_Point.getCoordinates()
    return getDistance(self.x, self.y, other_x, other_y, pixel_to_inch)
  

def getCombinations(arr: list[Rat_Point], choose: int):
  combinations = []
  for comb in itertools.combinations(arr, choose):
    combinations.append(comb)
  
  return combinations
  

def isACluster(arr: list[Rat_Point], threshold: int, pixel_to_inch: float) -> bool:
  count = 0
  sum_x = 0
  sum_y = 0
  for rat in arr:
    count += 1
    x, y = rat.getCoordinates()
    sum_x += x
    sum_y += y
  
  avg_x = sum_x / count
  avg_y = sum_y / count
  
  for rat in arr:
    rat_x, rat_y = rat.getCoordinates()
    dist = getDistance(rat_x, rat_y, avg_x, avg_y, pixel_to_inch)
    if dist > threshold:
      # print("not a cluster, distance: {} > threshold {}".format(dist, threshold))
      return False

  return True
    

# for a given rat i, returns (num_individuals - 1) dataframes, each one is the interaction of 
# rat i on rat j
def trackRatX(approach_distance: float, approach_time: float, null_frame_tolerance: int, i: int, individual_pd, frames_to_skip: int, frames_per_sec: int, rat_path: str, pixel_to_inch: float, titles: list):
  # print('I am number %d in process %d' % (i, os.getpid()))

  # how many frames the rat should be < 'approach_distance' from another to count as an approach behaviours
  approach_frames = round(approach_time * frames_per_sec) 
  
  
  # each element in approach_singular is a DataFrame of interactions with the other rats
  approach_singular = []
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
          engagement_frames = 0 # number of frames 
          null_frames = 0
          # go through the whole video comparing rat i with rat j
          while frame < (len(individual_pd[i])):
              nose_x, nose_y = individual_pd[i].iloc[frame, :2]
              while (math.isnan(nose_x)):
                  frame += 1
                  nose_x, nose_y = individual_pd[i].iloc[frame, :2]
              
              bodyparts = individual_pd[j].iloc[frame, :]
              min_index, min_distance, min_list = findMinDistance(bodyparts, nose_x, nose_y, pixel_to_inch)
              
              minute = ( (frame + frames_to_skip) / frames_per_sec) / 60
              if (not min_index is None and min_distance < approach_distance):
                engagement_frames += 1
                if (engagement_frames >= approach_frames):
                  # add the time stamp to the last column and append the row to the dataframe
                  min_list[-1] = minute
                  
                  # for debugging keep a list of interactions and the frames they happen
                  adjusted_frame = frame + frames_to_skip - approach_frames
                  saved_frames.append(adjusted_frame)
                  debugging_approaches.append('rat{}_to_{}'.format(i, titles[min_index//2]))
                  
                  approach_to_j.loc[len(approach_to_j.index)] = min_list
                  
                  engagement_frames = 0
                
              elif (min_index is None and null_frames < null_frame_tolerance):
                null_frames += 1
  
              else:
                engagement_frames = 0
              
              
              frame += 1
          
          approach_singular.append(approach_to_j)
          # save csv
          # approach_to_j.to_csv(path_or_buf = "C:/Users/micha/MonfilsLab/analysis/csv/rat{}_to_rat{}.csv".format(i, j))
      else:
          approach_singular.append(['same animal'])

      filename = "rat{}_to_rat{}.csv".format(i, j)
      path = os.path.join(path, filename)
      approach_to_j.to_csv(path_or_buf=path)
       
  return saved_frames, debugging_approaches
  
# from a list of x and y opints for body parts, return the min and its index
def findMinDistance(bodyparts: list, nose_x: float, nose_y: float, pixel_to_inch: float):
    min_index = -1
    min_distance = 1_000
    
    i = 0
    min_list = [None] * (round(len(bodyparts) / 2) + 1)
    
    while (i < len(bodyparts)):
        part_x = bodyparts[i]
        part_y = bodyparts[i + 1]
        
        if (not part_x is None):
            distance = getDistance(nose_x, nose_y, part_x, part_y, pixel_to_inch)
            if (distance < min_distance):
                min_distance = distance
                min_index = i
            
        i += 2
    
    if (min_index == -1):
        return None, None, min_list
    min_list[round(min_index/2)] = min_distance
    return min_index, min_distance, min_list
  
      
# returns the distance between two points (inches)
def getDistance(x1, y1, x2, y2, pixel_to_inch):
    #convert pixel distance to real world
    return convertPixelToInch(math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2)), pixel_to_inch)

# converts a pixel length to inches
def convertPixelToInch(x, pixel_to_inch: float):
  return x / pixel_to_inch

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
        pos_x = individual_pd[i].iloc[frame,j]

        if (not math.isnan(pos_x)):
          sum_x += pos_x
          count += 1
      else:
        pos_y = individual_pd[i].iloc[frame,j]
          
        if (not math.isnan(pos_y)):
          sum_y += pos_y
          count += 1
    
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


def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))
  
def printTime(frame: int) -> None:
  sec = frame // 60 % 60
  min = frame // 3600
  print(f'Frame: {frame}, Minute: {min}, Second: {sec}')
  

if __name__ == "__main__":
 main()
