import bpy
import numpy as np
from mathutils import Vector 

data = np.load('SFT_with_CNN/src/results/test2017.npz')
i = 18
gt = data['gt'][i].reshape((1002, 3))
pred = data['pred'][i].reshape((1002, 3))
for v in bpy.data.objects['pillow_2k'].data.vertices:
    v.co = Vector(gt[v.index,:])
for v in bpy.data.objects['pillow_2k.001'].data.vertices:
    v.co = Vector(pred[v.index,:])

