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
from scipy.misc import imread, imsave, imresize

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
    
#%%
dr = '/home/arvardaz/SFT_with_CNN/3D_models/paper/'
#dr = '/home/arvardaz/SFT_with_CNN/thesis/american_pillow_gt/'
outdir = '/home/arvardaz/Dropbox/datasets/paper_rig_REAL/'

cam_tags = read_xml(dr+'PS_Cameras.xml')
#cam_tags = read_xml(dr+'PSCamera.xml')
i = 0
for cam_tag in cam_tags:
#    if '0448' not in cam_tag['label']:
#        continue
    
    if cam_tag['id'] == '1':
        continue
    if cam_tag['id'] == '5':
        continue
    if int(cam_tag['id']) >= 11:
        continue
    name = cam_tag['label']
    newname = name[:-3]+'png'
    
    P = np.genfromtxt(dr+'paper_2k_coords.csv',delimiter=',')
#    P = np.genfromtxt(dr+'pillow_2k_coords.csv')

    P = P.reshape((1002,3))
    Ph = np.hstack((P, np.ones((1002, 1))))
   
    
#    K = np.array([[1887.3979937413362, 0, 1187.4168448401772-400],[0,1887.3979937413362,807.75879695084984],[0,0,1]])
#    K = np.array([[1887.3979937413362, 0, 1184.1541318707452-400],[0,1887.3979937413362,799.25775554378492],[0,0,1]])
#    K_out = np.array([[1887.3979937413362/10.714285714285714, 0, 112],[0,1887.3979937413362/10.714285714285714,112],[0,0,1]])
    K_out = np.array([[1878.8966522309493/10.714285714285714, 0, 112],[0,1878.8966522309493/10.714285714285714,112],[0,0,1]])
    
#    K = np.array([[1887.3979937413362, 0, 1184.1541318707452],[0,1887.3979937413362,799.25775554378492],[0,0,1]])
    
#    K = np.array([[1887.3979937413362/7.142857142857143, 0, 112],[0,1887.3979937413362/7.142857142857143,112],[0,0,1]])
    
#    f = 1600.0/224
    scale = 903.34751325
    RT = np.asarray(cam_tag.find('transform').text.split(), dtype=float).reshape((4,4))
    RT = RT[:3,:]
    R = RT[:3,:3].transpose()
    
    T = -np.matmul(R, RT[:,3]).reshape((3,1))
    
    RT_inv = np.hstack((R, T))
    
    
    p_cam = np.matmul(RT_inv, Ph.transpose()).transpose()
    p_cam = p_cam*scale/1000.0
    p_cam_flat  = p_cam.flatten()
    
    data = np.concatenate(([newname],[K_out[0,0]],[1.0], p_cam_flat))
    
    np.savetxt(outdir+newname[:-3]+'csv', data.reshape(1, data.shape[0]), delimiter=",", fmt="%s")
    
    img = plt.imread(dr+name)
#    imout = img[:,400:-400]
    imout = 255*np.ones((2400, 2400, 3))
    imout[400:-400] = img
    imout = imresize(imout,(224, 224))
    
    imsave(outdir+newname, imout)
        
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.set_aspect('equal')
#    ax.scatter(p_cam[:,0], p_cam[:,1], p_cam[:,2],c='r')
#    ax.scatter(0, 0, 0, c='c', marker='o')
#    ax.set_title(name)


#    im0 = p_cam[:,:3] / np.repeat(p_cam[:,2].reshape(1002,1),3,axis=1)
#    im = np.matmul(K_out, im0.transpose()).transpose()
#    fig, ax = plt.subplots()    
#    ax.imshow(imout)
#    ax.plot(im[:,0], im[:,1], 'g.',linewidth=0.01)
#    ax.set_title(name)
    
    
    
#Matrix(((-0.441739022731781, -0.012142248451709747, 0.02123897336423397, -0.03159768134355545),
#    (0.014909914694726467, 0.17142686247825623, 0.40876665711402893, -0.3030130863189697),
#    (-0.019310954958200455, 0.41011252999305725, -0.17023545503616333, 3.1456804275512695),
#    (0.0, 0.0, 0.0, 1.0)))

#scale = 0.44333333
#802.2556391147041