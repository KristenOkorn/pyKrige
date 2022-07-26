# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:08:37 2022

@author: okorn
"""

#PyKrige - 2nd attempt

#V2 has user put in endpoints of landfill instead of auto-generating grid

#Install the packages we need via COMMAND LINE - not python
    #pip install pykrige
    #conda install -c conda-forge pykrige

#Kriging algorithms to test:
    #OrdinaryKriging3D
    #UniversalKriging3D - KED is a particular case of this
    
#USER INPUT - what size grid do you want between your points?
user_grid = {
    'X' : 10,
    'Y' : 10,
    'Z' : 10
}

#USER INPUT - enter desired kriging (landfill) boundaries
#using entire "square" boxed off by roadways for XY
#using min & max ground height for Z
grid_bounds = {
    'min_lat' : 40.5819,
    'max_lat' : 40.595989,
    'min_long' : -104.831731,
    'max_long' : -104.811817,
    'min_elev' : 5052,
    'max_elev' : 5141
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
import functools

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

#create a grid to match the size of our XYZ data
#add 2 extra grid cells to each end as cushion
grid_x = np.arange(grid_bounds['min_long'],grid_bounds['max_long'],user_grid['X'])
grid_y = np.arange(grid_bounds['min_lat'],grid_bounds['max_lat'],user_grid['Y'])
grid_z = np.arange(grid_bounds['min_elev'],grid_bounds['max_elev'],user_grid['Z'])

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
    
    #pykrige doesn't understand NaN's so we need to replace them
    #use the median at this timestamp across all pods
    
    #first see if we have any Nan's
    if any(pd.isna(temp_val) == 1):
    
        #Find which indices we need to replace (as an array)
        inds = np.asarray(np.where(np.isnan(temp_val)))

        #Place column means in the indices. Align the arrays using take
        temp_x[inds] = np.nanmedian(temp_x)
        temp_y[inds] = np.nanmedian(temp_y)
        temp_z[inds] = np.nanmedian(temp_z)
    
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

    #Save frame if desired - will clutter folder space
    #plt.savefig(filename)
    #plt.close()
    
#Build final gif
with imageio.get_writer('Pykrige_V2_shortened_w_NANs.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
#Remove leftover files
for filename in set(filenames):
    os.remove(filename)