import bpy
import mathutils
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
    obj.rotation_euler = mathutils.Euler((np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)))
    
    #Translation
    obj_x = std * np.random.randn()
    obj_y = std * np.random.randn()
    obj_z = std * np.random.randn()
    obj.location = mathutils.Vector((obj_x, obj_y, obj_z))

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

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))


###############################################################################

directory = os.path.dirname(os.path.realpath(sys.argv[0])) + '/SFT_with_CNN/'


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


#Import object
bpy.ops.import_scene.obj(filepath = directory + '3D_models/American_pillow/3d_decimated_norm/pillow_2k.obj')
#bpy.ops.import_scene.obj(filepath='/home/arvardaz/SFT_with_CNN/3D_models/big_ball/3d_decimated/big_ball_2k.obj')

#Find object in scene
scene = bpy.context.scene
for ob in scene.objects:
    if not('Camera' in ob.name or 'Plane' in ob.name):
        obj_name = ob.name
obj = bpy.data.objects[obj_name]

#Select object
bpy.ops.object.select_all(action='DESELECT')
obj.select = True

#Camera object
cam = bpy.data.objects["Camera"]

#Import texture
bpy.ops.xps_tools.convert_to_cycles_selected()

# # Clear directory 'output'
#if path.exists(directory+ 'output'):
#    rmtree(directory + 'output')
#    makedirs(directory + 'output')

#with Timer():

    
iters = 3000
n_vert = len(obj.data.vertices)

ptCloudArr = np.empty((iters, n_vert*3))

fname = str(time()) #obj_name + 
for i in np.arange(iters):
    # Assign random poses to object
    randomRotateTranslate(obj, 3)

    # Save Image
    bpy.context.scene.render.filepath = directory+'output/{}_{:03}.png'.format(fname, i)
    #224x224
    bpy.context.scene.render.resolution_x = 448 
    bpy.context.scene.render.resolution_y = 448
    bpy.ops.render.render( write_still=True) 
    
    # Save point cloud to .csv
    saveMeshAsCloud(obj, cam, directory+'output/{}_{:03}.csv'.format(fname, i), False)

    #Store point cloud in array
#    ptCloudArr[i,:] = getCloud(obj)

#Save pt array to file
#np.savetxt(fname=directory+'output1/points_data.csv', X=ptCloudArr, delimiter=',')
    
    
#    bpy.ops.export_scene.obj(filepath='/home/arvardaz/SFT_with_CNN/output/{}_{:03}.obj'.format(obj_name, i), use_selection = True)


#Center the origin of object
#bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
#obj.location = mathutils.Vector((0,0,0))

##Track object with camera
#bpy.ops.object.select_all(action = "DESELECT")
#bpy.data.objects['Camera'].select = True
#obj.select = True
#bpy.context.scene.objects.active = obj
#bpy.ops.object.track_set(type = "TRACKTO")

#scene = bpy.context.scene
#for ob in scene.objects:
#    ob.select = True
#    bpy.ops.object.track_clear(type='CLEAR')
##    if ob.name == "Camera" or ob.name == 'Lamp':
#    ob.select = False

