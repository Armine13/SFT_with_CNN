import bpy
import mathutils
import numpy as np
import time
from os import makedirs, path
from shutil import rmtree
import sys
from glob import glob1
from time import time
#######################
#!/usr/bin/python

def randomRotateTranslate(meshObj, std_frwd, std_sdw):
    """Applies random rotation and translation to the given blender object.
    Translation is sampled from a normal distribution around the origin with given
    std. """
    
    #Rotation
    alpha = np.random.uniform(0, 360)
    beta = np.random.uniform(0, 360)
    gamma = np.random.uniform(0, 360)
    obj.rotation_euler = mathutils.Euler((np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)))
    
    #Translation
    obj_x = std_sdw * np.random.randn()
    obj_y = std_frwd * np.random.randn()
    obj_z = std_sdw * np.random.randn()
    obj.location = mathutils.Vector((obj_x, obj_y, obj_z))

#def saveEdgesLen(meshObj, points):
#    faces = np.empty((len(meshObj.data.polygons), 3), dtype=int)
#    for i, f in enumerate(meshObj.data.polygons):
#        faces[i, :] = np.asarray(f.vertices)
#      
#    edges = np.vstack((faces[:,:2], faces[:,1:], faces[:,[0,2]]))
#    edges = np.sort(edges, 1)
#    edges = np.vstack({tuple(row) for row in edges})
#        
#    dist = np.zeros(len(edges))
#    
#    dist = np.sqrt(np.sum(np.square(points[edges[:,0]] - points[edges[:,1]]),1))
#    
#    np.savetxt("edges.csv", edges)
#    np.savetxt("dist.csv", dist)
def randomFocalLength(camObj, mu, std):
    #change focal length
    camObj.lens = std * np.random.randn() + mu

def saveMeshAsCloud(meshObj, camObj, filename):
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
    
    i = mathutils.Matrix(np.eye(3)*[1,-1,-1])
    
    points = np.empty((n_vert, 3))    
    for vertex in meshObj.data.vertices:
        global_co =  meshObj.matrix_world * vertex.co #Object to world frame
        cam_co = worldCamMat * global_co #World to Camera frame
        cam_co = i * cam_co #Transform from left hand to right hand coords
        points[vertex.index,:] = cam_co.to_tuple() 
    
    points = points.flatten()
    #    np.savetxt(filename, points.reshape(1, points.shape[0]),delimiter=",")
    
    data = np.concatenate(([filename[:-3] + 'png'], points))
    np.savetxt(filename, data.reshape(1, data.shape[0]), delimiter=",", fmt="%s")


def getCloud(meshObj):
    """ Takes a blender object and returns the corresponding
    point cloud as a 1D numpy array"""
    
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

def setBackgroundImage(image_path, planeObj):
    mat_name = "myMaterial"
    mat = (bpy.data.materials.get(mat_name) or
           bpy.data.materials.new(mat_name))
    
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    
    # clear
    while(nodes): nodes.remove(nodes[0])
    
    output  = nodes.new("ShaderNodeOutputMaterial")
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    texture = nodes.new("ShaderNodeTexImage")
    uvmap   = nodes.new("ShaderNodeUVMap")
    
    texture.image = bpy.data.images.load(image_path)
    uvmap.uv_map = "UV"
    
    links.new( output.inputs['Surface'], diffuse.outputs['BSDF'])
    links.new(diffuse.inputs['Color'],   texture.outputs['Color'])
    
    planeObj.data.materials.append(mat)
    planeObj.active_material = mat
    

    
    bpy.ops.object.material_slot_remove() 
    
                    
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

#working directory
directory = os.path.dirname(os.path.realpath(sys.argv[0])) + '/SFT_with_CNN/'
#directory = '/home/arvardaz/SFT_with_CNN/'

#background
bg_dir = directory + 'SBU-RwC90/mixed/'
bg_list = glob1(bg_dir, '*.jpg')

## Camera object ##############################################################
cam = bpy.data.objects["Camera"]
cam.location = mathutils.Vector((0, 10, 0))
look_at(cam, mathutils.Vector((0,0,0)))

## Import object ##############################################################
bpy.ops.import_scene.obj(filepath = directory + '3D_models/American_pillow/3d_decimated/pillow_2k.obj')

#Find object in scene
scene = bpy.context.scene
for ob in scene.objects:
    if not('Camera' in ob.name or 'Plane' in ob.name):
        obj_name = ob.name
obj = bpy.data.objects[obj_name]

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

##  Clear directory 'output3'
#if path.exists(directory+ 'output3'):
#    rmtree(directory + 'output3')
#    makedirs(directory + 'output3')


## Loop #######################################################################
iters = 1
bg_plane = bpy.data.objects['Plane.001']
n_vert = len(obj.data.vertices)
ptCloudArr = np.empty((iters, n_vert*3))
fname = str(time()) #obj_name + 

           
bpy.ops.object.select_all(action='DESELECT')
bg_plane.select = True


for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for region in area.regions:
            if region.type == 'WINDOW':
                override = {'area': area, 'region': region, 'edit_object': bpy.context.edit_object}
                bpy.ops.uv.smart_project(override)
                
for i in np.arange(iters):
    # Assign random poses to object
    randomRotateTranslate(obj, 4, 1.5)
    randomFocalLength(bpy.data.cameras['Camera'], 35, 6.7)
    
    #select and load background image
    im_idx = np.random.choice(len(bg_list))
    setBackgroundImage(bg_dir + bg_list[im_idx], bg_plane)

    # Save Image
    bpy.context.scene.render.filepath = directory+'output3/{}_{:03}.png'.format(fname, i)
    #224x224
    bpy.context.scene.render.resolution_x = 448 
    bpy.context.scene.render.resolution_y = 448
    bpy.ops.render.render( write_still=True) 
    
    # Save point cloud to .csv
    saveMeshAsCloud(obj, cam, directory+'output3/{}_{:03}.csv'.format(fname, i))




