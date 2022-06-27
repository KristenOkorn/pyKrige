# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 06:21:41 2022

@author: okorn
"""

#PyKrige - 1st attempt

#Install the packages we need via COMMAND LINE - not python
    #pip install pykrige
    #conda install -c conda-forge pykrige

#Kriging algorithms to test:
    #OrdinaryKriging3D
    #UniversalKriging3D - KED is a particular case of this
    
#USER INPUT - what size grid do you want?
user_grid = {
    'X' : 5,
    'Y' : 5,
    'Z' : 5
}

#Import in necessary packages
#on first run, need to run: pip install pykrige
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D
import numpy as np
from matplotlib import pyplot as plt
from tkinter.filedialog import askdirectory
import os
import pandas as pd
import imageio

### Step 0: have some data

#Prompt user to select folder for analysis
path = askdirectory(title='Select Folder for analysis').replace("/","\\")

#Get the list of files from this directory
from os import listdir
from os.path import isfile, join
fileList = [f for f in listdir(path) if isfile(join(path, f))]

#Check how many files we have to iterate over
numPods = len(fileList)

#create empty array to hold our pod list
podList = [None] * numPods

#get our pod list by removing "_Field.txt" from the list of names
for u in range(numPods):
    podList[u] = fileList[u].replace('_Field.txt', '')

#create a dictionary to hold our data from each pod
data_dict = {}

#loop through each of the files & extract only the columns we need:
  #podName, time, estimate, latitude, longitude & altitude only
  
for i in range(numPods):
    #Get the name of the pod we're looking at
    currentPod = podList[i]
    
    #Create full file path for reading file
    filePath = os.path.join(path, fileList[i])
    
    #Load in the data from text file
    temp = pd.read_csv(filePath, sep=',',usecols=['podName','time','estimate','latitude','longitude','elevation'])
    
    #Save this into our data dictionary
    data_dict['data_{}'.format(currentPod)] = temp
    
#concatenate all of our data horizontally
full_data = pd.concat(data_dict.values(),axis=1)

#get the extrema of our spatial data points
grid_extrema = {
    'max_x' : max(temp['longitude'].astype(str).astype(float)),
    'min_x' : min(temp['longitude'].astype(str).astype(float)),
    
    'max_y' : max(temp['latitude'].astype(str).astype(float)),
    'min_y' : min(temp['latitude'].astype(str).astype(float)),

    'max_z' : max(temp['elevation'].astype(str).astype(float)),
    'min_z' : min(temp['elevation'].astype(str).astype(float))
}

#also get our grid sizes based on user input
grid_sizes = {
    'x' : (grid_extrema['max_x'] - grid_extrema['min_x']) / user_grid['X'],
    'y' : (grid_extrema['max_y'] - grid_extrema['min_y']) / user_grid['Y'],
    'z' : (grid_extrema['max_z'] - grid_extrema['min_z']) / user_grid['Z'],
    }

#create a grid to match the size of our XYZ data
grid_x = np.arange(grid_extrema['min_x'],grid_extrema['max_x'],grid_sizes['x'])
grid_y = np.arange(grid_extrema['min_y'],grid_extrema['max_y'],grid_sizes['y'])
grid_z = np.arange(grid_extrema['min_z'],grid_extrema['max_z'],grid_sizes['z'])

#initialize list storage for each of our images
filenames = []

#do our analysis for each column of the dataframe
#re-creating format used in this example: https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/examples/02_kriging3D.html#sphx-glr-download-examples-02-kriging3d-py

for k in range(len(full_data)):
    #pull out just the row we're looking at
    temp = full_data.iloc[k]
    #get our "estimates" as the values & convert to numpy array
    temp_val = temp.loc['estimate']
    temp_val = temp_val.to_numpy().astype(float)
    #get our "x" (longitude) & convert to numpy array
    temp_x = temp.loc['longitude']
    temp_x = temp_x.to_numpy().astype(float)
    #get our "y" (latitude) & convert to numpy array
    temp_y = temp.loc['latitude']
    temp_y = temp_y.to_numpy().astype(float)
    #get our "z" (elevation) & convert to numpy array
    temp_z = temp.loc['elevation']
    temp_z = temp_z.to_numpy().astype(float)
    
    #now need to put this data in the format pykrige wants it in
    #data = np. array ([ x1, y1, z1, value1
                        #x2, y2, z2, value2
                        #x3, y3, z3, value3 ])
    
    data = np.transpose(np.stack((temp_x,temp_y,temp_z,temp_val))).astype(float)
    
    #Create the 3D OK object
    ok3d = OrdinaryKriging3D(
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model="linear"
    )
    #Solves for 3D kriged volume & variance
    k3d1, ss3d = ok3d.execute("grid", temp_x, temp_y, temp_z)
    
    #Plot the image for this timestamp
    plt.title('Ordinary Kriging')
    plt.imshow(k3d1[:, :, 0], origin="lower")
    
    #Create file name and append it to a list
    filename = f'{k}.png'
    filenames.append(filename)

    #Save frame
    plt.savefig(filename)
    plt.close()
    
#Build final gif
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
#Remove leftover files
for filename in set(filenames):
    os.remove(filename)