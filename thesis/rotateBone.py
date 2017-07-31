#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:08:43 2017

@author: arvardaz
"""
import bpy
from mathutils import Euler
import numpy as np


bpy.data.objects['Armature'].pose.bones['Bone.001'].rotation_quaternion = Euler((np.random.uniform(-180,180),np.random.uniform(-180,180),np.random.uniform(-180,180))).to_quaternion()