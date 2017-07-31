#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:09:10 2017

@author: arvardaz
"""
from glob import glob1
import numpy as np
import re
import os 

#p = '/home/arvardaz/Dropbox/datasets/wo_fl/'
#p = '/home/arvardaz/Dropbox/datasets/wo_fl/dataset_rt+fl/'
p = '/home/arvardaz/Dropbox/datasets/pillow/train/'
#paths = [p, p+'/test/',p+'/train/', p+'/val/']
#files =[[path+f for f in glob1(path, '*.csv')] for path in paths]
#files = np.asarray(files)
#files = np.concatenate(files).ravel()

#
#for f in files:
#    a = open(f, 'r+')
#    text = a.read()
#    text = re.sub(r"1887.3979937413362","264.2357191237871", text)
#    a.seek(0)
#    a.write(text)
#    a.truncate()
#    a.close()

l = []
for root, dirs, files in os.walk(p):
    
    for file in files:
        if file.endswith('.csv'):
            l.append(root+'/'+file)
            
#f = l[2]
for f in l:
    csv = np.genfromtxt(f, dtype=None)
    
    csv = np.asarray(str(csv).split(','))
    
    imname = csv[0]
    if csv.shape[0] >= 3009:
        continue
    
    fl = csv[1].astype(np.float64)
#    else:
#        fl = 0.0
    points = csv[-3006:].astype(np.float64)
    
#    points = np.reshape(points, (1002, 3))
    
    #from matplotlib import pyplot
    #pyplot.plot(points[:,0], points[:,1],'bx')
    
#    if points[0,2] < 0:
#        points[:,1] = -points[:,1]
#        points[:,2] = -points[:,2]
    points = points.flatten()
    data = np.concatenate(([imname],[fl],[1.0], points))
    np.savetxt(f, data.reshape(1, data.shape[0]), delimiter=",", fmt="%s")
