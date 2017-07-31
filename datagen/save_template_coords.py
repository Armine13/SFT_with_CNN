#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:27:06 2017

@author: arvardaz
"""

import bpy
import numpy as np
from mathutils import Matrix
meshObj = bpy.data.objects['blue_tube_2k.002']
camObj = bpy.data.objects['Camera']
path = '/home/arvardaz/SFT_with_CNN/3D_models/tube/extra/3d/2k/'
filename = 'tube_2k_coords.csv'

n_vert = len(meshObj.data.vertices)
points = np.empty((n_vert, 3))    

worldCamMat = camObj.matrix_world.inverted()

i = Matrix(np.eye(3)*[1,-1,-1])

points = np.empty((n_vert, 3))
for vertex in meshObj.data.vertices:
    cam_co = meshObj.matrix_world * vertex.co#World to Camera frame
    points[vertex.index,:] = cam_co.to_tuple() 

points = points.flatten()

data = points
np.savetxt(path+filename, data.reshape(1, data.shape[0]), delimiter=",", fmt="%s")