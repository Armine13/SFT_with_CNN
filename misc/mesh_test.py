#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:07:46 2017

@author: arvardaz
"""
from openmesh import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

obj_path1 = '/home/arvardaz/SFT_with_CNN/output/pillow_2k_000.obj'
obj_path2 = '/home/arvardaz/SFT_with_CNN/output/pillow_2k_001.obj'

mesh1 = openmesh.TriMesh()
mesh2 = openmesh.TriMesh()

openmesh.read_mesh(mesh1, obj_path1)
openmesh.read_mesh(mesh2, obj_path2)
#
k = 0
for fh in mesh1.vertices():
    print(fh.idx(), [mesh1.point(fh)[i] for i in range(3)])
    
    k+=1
    if k >= 10:
        break
    
#
#prop_handle = VPropHandle()
#mesh1.add_property(prop_handle, "cogs")
#
#for vh in mesh1.vertices():
#    cog = TriMesh.Point(0,0,0)
#    valence = 0
#    for neighbor in mesh1.vv(vh):
#        cog += mesh1.point(neighbor)
#        valence += 1
#        print(neighbor.idx())
#    print('\n')
#    mesh1.set_property(prop_handle, vh, cog / valence)