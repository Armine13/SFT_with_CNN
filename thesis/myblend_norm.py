import bpy
from mathutils import Vector, Matrix, Euler
import numpy as np
import time
from os import makedirs, path
import os
from shutil import rmtree
from glob import glob1
#from scipy.misc import imread, imsave
import sys
from time import time
#######################
#!/usr/bin/python

def randomRotateTranslate(meshObj, rad):
    """Applies random rotation and translation to the given blender object.
    Translation is sampled from a normal distribution around the origin with given
    std. """
    
    #Rotation
    alpha = np.random.uniform(0, 360)
    beta = np.random.uniform(0, 360)
    gamma = np.random.uniform(0, 360)
    obj.rotation_euler = Euler((np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)))
    
    #Translation
    obj_x = np.random.uniform(-rad, rad)
    obj_y = np.random.uniform(-rad, rad)
    obj_z = np.random.uniform(-rad, rad)
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
    
def saveMeshAsCloud(meshObj, camObj, filename, path, saveEdgesFlag = False):
    """ Takes a blender object and saves the corresponding
    point cloud to <filename.csv> file """
    n_vert = len(meshObj.data.vertices)
    points = np.empty((n_vert, 3))    
    
    worldCamMat = camObj.matrix_world.inverted()
    
    i = Matrix(np.eye(3)*[1,-1,-1])
    
    points = np.empty((n_vert, 3))    
    for vertex in meshObj.data.vertices:
        cam_co = worldCamMat *  meshObj.matrix_world * vertex.co#World to Camera frame
        cam_co = i * cam_co #Transform from left hand to right hand coords
        points[vertex.index,:] = cam_co.to_tuple() 
    
    points = points.flatten()
    
    data = np.concatenate(([filename[:-3] + 'png'], points))
    np.savetxt(path+filename, data.reshape(1, data.shape[0]), delimiter=",", fmt="%s")
  
    
def look_at(obj_camera, point):
    """ Points given camera in the direction of point """
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume ptCloudArrwe're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

def randomFocalLength(camObj, mn, mx):#mu, std):
    #change focal length
    camObj.lens = np.random.uniform(mn, mx) #std * np.random.randn() + mu

def randomLighting(mn,mx):
    bpy.data.objects['Sphere']#top
    bpy.data.objects['Sphere.001']#bottom
    bpy.data.objects['Sphere.002']#left
    bpy.data.objects['Sphere.003']#right
    ##Sphere 1 (top)
    obj = bpy.data.objects['Sphere']
    hide = np.random.randint(2)
    obj.hide = hide
    if not hide:
        a = np.random.uniform(-200, 200)
        b = np.random.uniform(-200, 200)
        obj.location = Vector((a, b, 200))    
    ##Sphere 2 (bottom)
    obj = bpy.data.objects['Sphere.001']
    hide = np.random.randint(2)
    obj.hide = hide
    if not hide:
        a = np.random.uniform(-200, 200)
        b = np.random.uniform(-200, 200)
        obj.location = Vector((a, b, -200))    	
    ##Sphere 2 (left)
    obj = bpy.data.objects['Sphere.002']
    hide = np.random.randint(2)
    obj.hide = hide
    if not hide:
        a = np.random.uniform(-200, 200)
        b = np.random.uniform(-200, 200)
        obj.location = Vector((200, a, b))    
    ##Sphere 2 (right)
    obj = bpy.data.objects['Sphere.003']
    hide = np.random.randint(2)
    obj.hide = hide
    if not hide:
        a = np.random.uniform(-200, 200)
        b = np.random.uniform(-200, 200)
        obj.location = Vector((-200, a, b))    
    bpy.data.materials['sphereMat'].node_tree.nodes['Emission'].inputs[1].default_value = np.random.uniform(mn, mx)
    bpy.data.materials['sphereMat.001'].node_tree.nodes['Emission'].inputs[1].default_value = np.random.uniform(mn, mx)
    bpy.data.materials['sphereMat.002'].node_tree.nodes['Emission'].inputs[1].default_value = np.random.uniform(mn, mx)
    bpy.data.materials['sphereMat.003'].node_tree.nodes['Emission'].inputs[1].default_value = np.random.uniform(mn, mx)
    for ob in bpy.context.scene.objects: ob.hide_render = ob.hide

def setBackgroundImage(image_path, planeObj):
    
#    bpy.ops.object.material_slot_remove() 
    
    bpy.data.objects['PlaneBG'].rotation_euler = (np.deg2rad(90), np.deg2rad(np.random.randint(0,4)*90), 0)
    
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
    
#    bpy.ops.object.material_slot_remove()                       

def unpackBackground(bg_plane):
#bg_plane = bpy.data.objects['PlaneBG']
    bpy.context.scene.objects.active = bg_plane
    bpy.ops.object.select_all(action='DESELECT')
    bg_plane.hide = False
    bg_plane.select = True    
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    override = {'area': area, 'region': region, 'edit_object': bpy.context.edit_object}
                    bpy.ops.uv.smart_project(override)
                    


###############################################################################

directory = os.path.dirname(os.path.realpath(sys.argv[0])) + '/SFT_with_CNN/'
outdir = directory + 'datasets/dataset_rt+fl+l+bg_uni/train/'
#outdir = directory + 'temp/'

#background
bg_dir = directory + 'SBU-RwC90/mixed/slices/'
bg_list = glob1(bg_dir, '*.jpg')

#directory = '/home/arvardaz/SFT_with_CNN/'

##Delete all existing objects from scene except Camera and Plane
scene = bpy.context.scene
#for ob in scene.objects:
#    ob.select = True
##    bpy.ops.object.track_clear(type='CLEAR')
#    if ob.name == "Camera" or ob.name == 'Plane':
#        ob.select = False
##        bpy.ops.object.track_clear(type='CLEAR')
#bpy.ops.object.delete()

## Camera object ##############################################################
cam = bpy.data.objects["Camera"]
cam.location = Vector((0, 40, 0))
look_at(cam, Vector((0,0,0)))

## Import object ##############################################################
bpy.ops.import_scene.obj(filepath = directory + '3D_models/American_pillow/3d_decimated_norm/pillow_2k.obj')


#Find object in scene
scene = bpy.context.scene
for ob in scene.objects:
    if not('Camera' in ob.name or 'Plane' in ob.name or 'Sphere' in ob.name):
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
iters = 30000

fname = str(time())  

planeBG = bpy.data.objects['PlaneBG']
planeBG.select = True
unpackBackground(planeBG)

#224x224
bpy.data.scenes['Scene'].render.resolution_percentage = 100
bpy.context.scene.render.resolution_x = 224
bpy.context.scene.render.resolution_y = 224

bpy.data.scenes['Scene'].cycles.samples = 800
              
for i in np.arange(iters):
    
    # Assign random poses to object
    randomRotateTranslate(obj, 8)#std=3
    
    randomFocalLength(bpy.data.cameras['Camera'], 20, 50)# 35, 6.7)
    
    randomLighting(15000, 55000)#35000, 10000)
    
    #select and load background image
    im_idx = np.random.choice(len(bg_list))
    setBackgroundImage(bg_dir + bg_list[im_idx], planeBG)
    # Save Image
    bpy.context.scene.render.filepath = outdir + '{}_{:04}.png'.format(fname, i)
    bpy.ops.render.render( write_still=True)
    # Save point cloud to .csv
    saveMeshAsCloud(obj, cam, '{}_{:04}.csv'.format(fname, i), outdir, False)
    



    
    
    
    
    