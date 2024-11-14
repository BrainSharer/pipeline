# Cell Detection Sudo-code

## Step1 segment tiff:
```
For Each channel:

​      For each section:

​            Split full resolution aligned image into 2*5 tiles.

​            if manual annotation exist in the data base for this brain

​                 Create a csv file with manual annotations on this section
```

## Step2 Create example:
```
For each section:

​    For each tile:

​           Subtract blurred image

​            Find connected segments

​            For each connected segment:

​                   if this segment is smaller than 100000 pixel in area and do not clip the tile borders:

​                       Crop out the image of the segment with a 80*80 box on CH1 and CH3

​                        Create an example with the coordinate of segment and images

​                        Label an example if a manual annotation lands on the segment

​        Save all Examples in that section as a pickle file
```
## Step3 Create Feature:
```
For each section:

​     load all examples of this section from pickle file

​     Calculate features for each section

​      Store all features of this section in a csv file
```
## Step4 Detect Cells:
```
pull all features from different sections to a combined features file and store as a csv

for each connected segment:

​       find the mean and std score from 30 models.

 Store mean and std score for all segments, along with their coordinates in a csv file
```


## This process is repeated for different thresholds

### Pulling result of different image thresholds
```
pull the detection score from all thresholds for an animal.

Group detection from different threshold by doing the following:

Calculating the distance matrix between all points

Get a list of paired points whose distance to each other is within threshold.  
If A is close to B there would be two entry: A,B and B,A

remove duplicate pairs, we should only have A,B now

group pairs into bigger groups: eg if we have pair A,B ; B,C; E,F we would have group: A,B,C and E,F

find out which category (threshold+sure/unsure) the points in each group belong to

Each group is treated as a detection.

For Each group:

​    if all detections in group are sures, this is a sure detection, and the coordinate from the 
threshold with highest mean score is selected for this detection

​    If all detection are unsure, the detection is marked unsure

​    If we have a mix of sure and unsures:

​            if sures> unsure:

​                  This is a sure point

​            if sures<=unsure:

​                   This is an unsure point

​    the coordinate from the threshold with highest mean score is selected for this detection

collect the category (sure/unsure) and coordinates(x,y,z) from each group and store in a csv 
```
