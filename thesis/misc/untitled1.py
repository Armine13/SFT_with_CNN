import bpy
import mathutils
import numpy as np
from os import makedirs, path
from shutil import rmtree
from glob import glob1

filename = 'SFT_with_CNN/src/filefileneq.csv'

rt = np.array([[-0.65432129, -0.45595637,  0.60329715, -4.32548582],
        [ 0.48081227,  0.36492345,  0.79727689, -4.79537776],
        [-0.58368075,  0.81174791, -0.01954776,  5.48092408],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])


camObj = bpy.data.objects['Camera']
meshObj = bpy.data.objects[3]

#camObj.matrix_world = Matrix(rt)
#bpy.ops.xps_tools.convert_to_cycles_selected()

i = mathutils.Matrix(np.eye(3)*[1,-1,-1])

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
    cam_co = cam_co * i
    points[vertex.index,:] = cam_co.to_tuple() 


points = points.flatten()

#    np.savetxt(filename, points.reshape(1, points.shape[0]),delimiter=",")

data = np.concatenate(([filename[:-3] + 'png'], points))
np.savetxt(filename, data.reshape(1, data.shape[0]), delimiter=",", fmt="%s")