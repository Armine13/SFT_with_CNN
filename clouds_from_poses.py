#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:51:05 2017

@author: arvardaz
"""

from bs4 import BeautifulSoup as BS
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def equal_axis(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def read_xml(path):        
    f = open(path)
    line = f.read()
    f.close()
    soup = BS(line)
    return soup.findAll('camera', {"label":True})
    
dr = '/home/arvardaz/SFT_with_CNN/american_pillow_gt/'
cam_tags = read_xml(dr+'PSCamera.xml')
i = 0
for cam_tag in cam_tags:
    if '0448' not in cam_tag['label']:
        continue
    
    if cam_tag['id'] == '50':
        continue
    
    
    name = cam_tag['label']
    
    P = np.genfromtxt(dr+'pillow_2k_coords.csv')
    P = P.reshape((1002,3))
    
    Ph = np.hstack((P, np.ones((1002, 1))))
    
    K = np.array([[1887.3979937413362, 0, 1187.4168448401772],[0,1887.3979937413362,807.75879695084984],[0,0,1]])
    
    RT = np.asarray(cam_tag.find('transform').text.split(), dtype=float).reshape((4,4))
    RT = RT[:3,:]
    R = RT[:3,:3].transpose()
    
    T = -np.matmul(R, RT[:,3]).reshape((3,1))
    
    RT_inv = np.hstack((R, T))
    
    p_cam = np.matmul(RT_inv, Ph.transpose()).transpose()
    
#    np.savetxt(dr+name[:-4]+'.csv', p_cam.flatten())
        
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.set_aspect('equal')
#    ax.scatter(p_cam[:,0], p_cam[:,1], p_cam[:,2],c='r')
#    ax.scatter(0, 0, 0, c='c', marker='o')
#    ax.set_title(name)
#   
#    im0 = p_cam[:,:3] / np.repeat(p_cam[:,2].reshape(1002,1),3,axis=1)
#    im = np.matmul(K, im0.transpose()).transpose()     
#    fig, ax = plt.subplots()
#    ax.imshow(plt.imread('/home/arvardaz/SFT_with_CNN/american_pillow_images/'+name))
#    ax.plot(im[:,0], im[:,1], 'rx')
#    ax.set_title(name)