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

def randomRotateTranslate(meshObj, s, d):
    """Applies random rotation and translation to the given blender object.
    Translation is sampled from a normal distribution around the origin with given
    std. """
    
    #Rotation
    alpha = np.random.uniform(0, 360)
    beta = np.random.uniform(0, 360)
    gamma = np.random.uniform(0, 360)
    meshObj.rotation_euler = Euler((np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)))
    
    #Translation
    obj_x = np.random.uniform(-s, s)
    obj_y = np.random.uniform(-d, d)
    obj_z = np.random.uniform(-s, s)
    meshObj.location = Vector((obj_x, obj_y, obj_z))

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
    
    K = get_calibration_matrix_K_from_blender(camObj.data)
    data = np.concatenate(([filename[:-3] + 'png'],[K[0,0]], points))
    #    data = np.concatenate(([filename[:-3] + 'png'], points))
    np.savetxt(path+filename, data.reshape(1, data.shape[0]), delimiter=",", fmt="%s")
  
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = np.array([[alpha_u, skew,u_0],
                  [0,  alpha_v, v_0],
                  [0,    0,      1 ]])
    K[1,1] = K[0,0]
    return K    

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
    hide_all=np.ones(4)
    hide_all[np.random.choice(range(4), np.random.randint(1, 5), False)] = 0
    
    obj = bpy.data.objects['Sphere']
    hide = hide_all[0]
    obj.hide = hide
    if not hide:
        a = np.random.uniform(-400, 400)
        b = np.random.uniform(-400, 400)
        obj.location = Vector((a, b, 400))    
    ##Sphere 2 (bottom)
    obj = bpy.data.objects['Sphere.001']
    hide = hide_all[1]
    obj.hide = hide
    if not hide:
        a = np.random.uniform(-400, 400)
        b = np.random.uniform(-400, 400)
        obj.location = Vector((a, b, -400))    	
    ##Sphere 2 (left)
    obj = bpy.data.objects['Sphere.002']
    hide = hide_all[2]
    obj.hide = hide
    if not hide:
        a = np.random.uniform(-400, 400)
        b = np.random.uniform(-400, 400)
        obj.location = Vector((400, a, b))    
    ##Sphere 2 (right)
    obj = bpy.data.objects['Sphere.003']
    hide = hide_all[3]
    obj.hide = hide
    if not hide:
        a = np.random.uniform(-400, 200)
        b = np.random.uniform(-400, 400)
        obj.location = Vector((-400, a, b))    
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
#    bg_plane = bpy.data.objects['PlaneBG']
#    bpy.context.scene.objects.active = bg_plane
#    bpy.ops.object.select_all(action='DESELECT')
#    bg_plane.hide = False
#    bg_plane.select = True   
#    
#    bg_plane = bpy.data.objects['PlaneBG']



    bpy.ops.object.select_all(action='DESELECT')
    bg_plane.hide = False
    bg_plane.select = True    
    bpy.context.scene.objects.active = bg_plane
    
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        for region in area.regions:
                            if region.type == 'WINDOW':
                                override = {'window': window, 'screen': screen, 'area': area, 'region': region}
#                                bpy.ops.uv.project_from_view(override, orthographic=False, correct_aspect=True, clip_to_bounds=False, scale_to_bounds=True)
                                bpy.ops.uv.smart_project(override)
#                            
 
def prepareCamera(loc):
    cam = bpy.data.objects["Camera"]
    cam.location = Vector(loc)
    look_at(cam, Vector((0,0,0)))
    return cam

def prepareObject():
    obj_name = 'pillow_2k'
    #    bpy.context.object.name = obj_name
    
    obj = bpy.data.objects['backup.001']
    obj.name = obj_name
    #    obj = bpy.data.objects[obj_name]
    obj.hide = False
    obj.hide_render = False
    bpy.context.scene.objects.active = obj
    #Select object
    obj_arm = obj.parent
    bpy.ops.object.select_all(action='DESELECT')
    obj_arm.select = True
    #Center the origin of object
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    obj_arm.location = Vector((0,0,0))
    # set object frame to world frame
#    obj.matrix_world.identity()
    #Import texture
    #bpy.context.scene.objects.active = obj
    #bpy.ops.xps_tools.convert_to_cycles_selected()
    return obj
def applyClothModifier(obj):
    bpy.context.scene.objects.active = obj

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                override = bpy.context.copy()
                override['area'] = area
                bpy.ops.object.modifier_apply(override, apply_as='DATA',modifier='Cloth')

def applyArmModifier(obj):
    bpy.context.scene.objects.active = obj

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                override = bpy.context.copy()
                override['area'] = area
                bpy.ops.object.modifier_apply(override, apply_as='DATA',modifier='Armature.003')
                
    
#    bpy.context.scene.objects.active = obj
#    bpy.ops.object.modifier_apply(apply_as='DATA',modifier='Cloth')
# this operator will 'work' or 'operate' on the active object we just set
#    bpy.ops.object.modifier_apply(modifier="Cloth")

#    bpy.ops.object.modiobj_armfier_apply(apply_as='DATA', modifier="Cloth")
#    
#    for window in bpy.context.window_manager.windows:
#        for area in window.screen.areas:
##        for area in bpy.context.screen.areas:
##            original_type = bpy.context.area.type
##            bpy.context.area.type = "VIEW_3D"
##            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
##            bpy.context.area.type = original_type
#            original_type = area.type
#            area.type = "VIEW_3D"
##            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
#            bpy.context.scene.objects.active = obj
#            obj.select = True
#            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Cloth")
#            area.type = original_type
#
##            
###############################################################################

directory = os.path.dirname(os.path.realpath(sys.argv[0])) + '/SFT_with_CNN/'
#outdir = directory + 'datasets/dataset_def_rt+fl+l+bg/train/'
#outdir = '/home/arvardaz/Dropbox/datasets/dataset_def_rt+fl+l+bg5/train/'
outdir = '/home/arvardaz/Dropbox/datasets/fl/rig_rt+fl/train/'
outdir_test = '/home/arvardaz/Dropbox/datasets/fl/rig_rt+fl/test/'
#outdir = directory + '/temp/'

#background
bg_dir = directory + 'SBU-RwC90/mixed/slices/'
bg_list = glob1(bg_dir, '*.jpg')

iters = 20000
#frame = 7




## Import object ##############################################################
#bpy.ops.import_scene.obj(filepath = directory + '3D_models/American_pillow/3d_decimated_norm/pillow_2k.obj')
#bpy.ops.wm.open_mainfile(filepath = '/home/arvardaz/SFT_with_CNN/deformation.blend')
bpy.ops.wm.open_mainfile(filepath = '/home/arvardaz/SFT_with_CNN/blender/def_rig_pillow.blend')

cam = prepareCamera((0, 30, 0)) #(0, 40, 0)
###############################################################################


filename = str(time())

planeBG = bpy.data.objects['PlaneBG']
planeBG.hide = True
planeBG.select = True
bpy.data.objects['Plane'].hide = False
bpy.data.objects['Plane'].hide_render = False
#unpackBackground(planeBG)

#quit()
#224x224
bpy.data.scenes['Scene'].render.resolution_percentage = 100
bpy.context.scene.render.resolution_x = 224
bpy.context.scene.render.resolution_y = 224

bpy.data.scenes['Scene'].cycles.samples = 800
all_objects = bpy.data.objects
obj_backup = bpy.data.objects['backup']

for i in np.arange(iters):


    bpy.data.objects['backup'].select = True
    #obj_backup.hide = False
    bpy.context.scene.objects.active = obj_backup
    obj = obj_backup.copy() # duplicate linked
    obj.data = obj_backup.data.copy() # optional: make this a real duplicate (not linked)
    obj.name = 'pillow_2k'
    obj.hide = False
    obj.hide_render = False
    bpy.context.scene.objects.link(obj) # add to scene

#    obj.parent.pose.bones['Bone.001'].rotation_quaternion = Euler((np.random.uniform(-180,180),np.random.uniform(-180,180),np.random.uniform(-180,180))).to_quaternion()
    
    applyArmModifier(obj)
    
    bpy.data.objects['pillow_2k'].parent = None
    obj.location = Vector((0,0,0))

    # add one of these functions to frame_change_pre handler:

    
    # Assign random poses to object
#    randomRotateTranslate(obj, 8)#std=3
    randomRotateTranslate(obj, 4, 10)#std=3
    
    
    randomFocalLength(bpy.data.cameras['Camera'], 20, 50)# 35, 6.7)
    
#    randomLighting(20000,60000)#(35000, 35000)#35000, 10000)
    
    #select and load background image
#    im_idx = np.random.choice(len(bg_list))
#    setBackgroundImage(bg_dir + bg_list[im_idx], planeBG)
    
    
#    obj = prepareObject()

    #Move force
#    bpy.data.objects['Field'].constraints['Limit Distance'].target = obj
#    bpy.data.objects['Field'].location = obj.location + Vector((np.random.uniform(-10, 10), np.random.uniform(0, 10), np.random.uniform(-10, 10)))
    #Add deformation
#    bpy.context.scene.frame_set(frame)
#    bpy.data.scenes['Scene'].frame_current=frame
#    bpy.ops.ptcache.bake_all(bake=False)
    # Save Image
    
    
#    applyClothModifier(obj)
    
    if i % 10 != 0:
        bpy.context.scene.render.filepath = outdir + '{}_{:04}.png'.format(filename, i)
        bpy.ops.render.render(write_still=True)
        saveMeshAsCloud(obj, cam, '{}_{:04}.csv'.format(filename, i), outdir, False)
    else:
        bpy.context.scene.render.filepath = outdir_test + '{}_{:04}.png'.format(filename, i)
        bpy.ops.render.render(write_still=True)
        saveMeshAsCloud(obj, cam, '{}_{:04}.csv'.format(filename, i), outdir_test, False)
    
#    bpy.ops.ptcache.free_bake_all()
    
    
    all_objects.remove(all_objects["pillow_2k"], True)
#    all_objects.remove(all_objects["Armature.001"], True)
    





    
    
    
    
