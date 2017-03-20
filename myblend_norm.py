import bpy
from mathutils import Vector, Matrix, Euler
import numpy as np
import time
from os import makedirs, path
from shutil import rmtree
import sys
from time import time
#######################
#!/usr/bin/python

def randomRotateTranslate(meshObj, std):
    """Applies random rotation and translation to the given blender object.
    Translation is sampled from a normal distribution around the origin with given
    std. """
    
    #Rotation
    alpha = np.random.uniform(0, 360)
    beta = np.random.uniform(0, 360)
    gamma = np.random.uniform(0, 360)
    obj.rotation_euler = Euler((np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)))
    
    #Translation
    obj_x = std * np.random.randn()
    obj_y = std * np.random.randn()
    obj_z = std * np.random.randn()
    obj.location = Vector((obj_x, obj_y, obj_z))

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
    
def saveMeshAsCloud(meshObj, camObj, filename, saveEdgesFlag = False):
    """ Takes a blender object and saves the corresponding
    point cloud to <filename.csv> file """
    
    n_vert = len(meshObj.data.vertices)
    points = np.empty((n_vert, 3))    
    
#    camWorldMat = np.asarray(camObj.matrix_world)
    #Invert [Rt]
    worldCamMat = camObj.matrix_world.inverted()
    
    worldCamMat = Matrix(worldCamMat) #from numpy to Matrix
        
    points = np.empty((n_vert, 3))    
    for vertex in meshObj.data.vertices:
        cam_co = worldCamMat * meshObj.matrix_world * vertex.co 
        points[vertex.index,:] = cam_co.to_tuple() 
    
    if saveEdgesFlag == True:
        saveEdgesLen(meshObj, points)
        
    points = points.flatten()
    
#    np.savetxt(filename, points.reshape(1, points.shape[0]),delimiter=",")
    
    data = np.concatenate(([filename[:-3] + 'png'], points))
    np.savetxt(filename, data.reshape(1, data.shape[0]), delimiter=",", fmt="%s")


def getCloud(meshObj):
    """ Takes a blender object and returns the corresponding
    point cloud to as a 1D numpy array"""
    
    n_vert = len(meshObj.data.vertices)
    points = np.empty((n_vert, 3))    
    for vertex in meshObj.data.vertices:
        global_co = meshObj.matrix_world * vertex.co #Covert object coords to global
        points[vertex.index,:] = global_co.to_tuple()     
    points = points.flatten()
    return points
    
def look_at(obj_camera, point):
    """ Points given camera in the direction of point """
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume ptCloudArrwe're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

def randomFocalLength(camObj, mu, std):
    #change focal length
    camObj.lens = std * np.random.randn() + mu

###############################################################################

directory = os.path.dirname(os.path.realpath(sys.argv[0])) + '/SFT_with_CNN/'
outdir = directory + 'dataset_rt+fl/'

#directory = '/home/arvardaz/SFT_with_CNN/'

##Delete all existing objects from scene except Camera and Plane
scene = bpy.context.scene
for ob in scene.objects:
    ob.select = True
#    bpy.ops.object.track_clear(type='CLEAR')
    if ob.name == "Camera" or ob.name == 'Plane':
        ob.select = False
#        bpy.ops.object.track_clear(type='CLEAR')
bpy.ops.object.delete()

## Camera object ##############################################################
cam = bpy.data.objects["Camera"]
cam.location = Vector((0, 40, 0))
look_at(cam, Vector((0,0,0)))

## Import object ##############################################################
bpy.ops.import_scene.obj(filepath = directory + '3D_models/American_pillow/3d_decimated_norm/pillow_2k.obj')


#Find object in scene
scene = bpy.context.scene
for ob in scene.objects:
    if not('Camera' in ob.name or 'Plane' in ob.name):
        obj_name = ob.name
obj = bpy.data.objects[obj_name]
#
#Select object
bpy.ops.object.select_all(action='DESELECT')
obj.select = True
#Center the origin of object
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
# set object frame to world frame
obj.matrix_world.identity()
#Import texture
bpy.ops.xps_tools.convert_to_cycles_selected()


###############################################################################

#  Clear directory 'output3'
#if path.exists(directory+ 'output3'):
#    rmtree(directory + 'output3')
#    makedirs(directory + 'output3')

## Loop #######################################################################
iters = 9000
n_vert = len(obj.data.vertices)

ptCloudArr = np.empty((iters, n_vert*3))

fname = str(time()) #obj_name + 

#224x224
bpy.data.scenes['Scene'].render.resolution_percentage = 100
bpy.context.scene.render.resolution_x = 224
bpy.context.scene.render.resolution_y = 224
for i in np.arange(iters):
    # Assign random poses to object
    randomRotateTranslate(obj, 3)
    
    randomFocalLength(bpy.data.cameras['Camera'], 35, 6.7)
    # Save Image
    bpy.context.scene.render.filepath = outdir + '{}_{:04}.png'.format(fname, i)
    

    bpy.ops.render.render( write_still=True)
    
    # Save point cloud to .csv
    saveMeshAsCloud(obj, cam, outdir + '{}_{:04}.csv'.format(fname, i), False)

#filename = os.path.join(os.path.basename(bpy.data.filepath), "/home/arvardaz/SFT_with_CNN/getCalibMatBlender.py")
#exec(compile(open(filename).read(), filename, 'exec'))