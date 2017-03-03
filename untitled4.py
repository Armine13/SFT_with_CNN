#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:14:45 2017

@author: arvardaz
"""

import mathutils
import numpy as np

meshObj = bpy.data.objects[3]
camObj = bpy.data.objects["Camera"]
n_vert = len(meshObj.data.vertices)
points = np.empty((n_vert, 3))    

camWorldMat = np.asarray(camObj.matrix_world)
#Invert [Rt]
worldCamMat = np.vstack((np.hstack((camWorldMat[:3,:3].transpose(), 
                                    -np.dot(camWorldMat[:3,:3].transpose(), 
                                            camWorldMat[:3,3].reshape(3,1)))), [0,0,0,1]))

worldCamMat = mathutils.Matrix(worldCamMat) #from numpy to Matrix
    
points = np.empty((n_vert, 3))    
for vertex in meshObj.data.vertices:
    global_co =  meshObj.matrix_world * vertex.co #Object to world frame
    cam_co = worldCamMat * global_co #World to Camera frame
    points[vertex.index,:] = cam_co.to_tuple() 
    
def saveEdgesLen(meshObj, points):
    faces = np.empty((len(meshObj.data.polygons), 3), dtype=int)
    for i, f in enumerate(meshObj.data.polygons):
        faces[i, :] = np.asarray(f.vertices)
      
    edges = np.vstack((faces[:,:2], faces[:,1:], faces[:,[0,2]]))
    edges = np.sort(edges, 1)
    edges = np.vstack({tuple(row) for row in edges})
        
    dist = np.zeros(len(edges))
    
    dist = np.sqrt(np.sum(np.square(points[edges[:,0]] - points[edges[:,1]]),1))
    
    np.savetxt("edges.csv", edges)
    np.savetxt("dist.csv", dist)