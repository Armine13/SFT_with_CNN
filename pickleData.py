#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:22:22 2017

@author: arvardaz
"""

import os
import numpy as np
from scipy import ndimage

directory = os.path.join("/home/arvardaz/SFT_with_CNN","output")

#CSV
n = 1002*3 #dimension of data

data = np.empty((0,n),'float32')
for root,dirs,files in os.walk(directory):
    #print(root, dirs, files)
    for file in files:
        if file.endswith(".csv"):
            
            file_name = os.path.join(directory, file)
            try:
                d = np.genfromtxt(file_name, dtype='float32')
            except IOError as e:
                print('Could not read:', file_name, ':', e, ' - skipping.')
                continue
            d = d.flatten()
            data = np.vstack([data, d])               
np.save(file=os.path.join(directory, 'data_csv.npy'), arr=data, allow_pickle=True)


##PNG
#n = 224 * 224
#data = np.empty((0,n),'float32')
#for root,dirs,files in os.walk(directory):
#    #print(root, dirs, files)
#    for file in files:
#        if file.endswith(".png"):
#            
#            file_name = os.path.join(directory, file)
#            try:
#                im = ndimage.imread(file_name).astype(float)
#            except IOError as e:
#                print('Could not read:', file_name, ':', e, ' - skipping.')
#                continue
#            #d = d.flatten()
#            data = np.vstack([data, d])
#np.save(file=os.path.join(directory, 'data_csv.npy'), arr=data, allow_pickle=True)
