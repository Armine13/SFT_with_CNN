import bpy
import mathutils
import numpy as np

#!/usr/bin/python
def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()
    
# Setting up the scene ######################################################

bpy.ops.wm.read_homefile()

###Delete all existing objects except Camera and Lamp
#scene = bpy.context.scene
#for ob in scene.objects:
#    ob.select = True
#    bpy.ops.object.track_clear(type='CLEAR')
##    if ob.name == "Camera" or ob.name == 'Lamp':
#    ob.select = False
##        bpy.ops.object.track_clear(type='CLEAR')
###bpy.ops.object.delete()


#scene = bpy.context.scene
# Create new lamp datablock
#lamp_data = bpy.data.lamps.new(name="new_lamp", type='POINT')

# Create new object with our lamp datablock
#lamp_object = bpy.data.objects.new(name="new_lamp", object_data=lamp_data)

# Link lamp object to the scene so it'll appear in this scene
#scene.objects.link(lamp_object)

# Place lamp to a specified location
#lamp_object.location = (5.0, 5.0, 5.0)



#Import object and 
bpy.ops.import_scene.obj(filepath='/home/arvardaz/SFT_with_CNN/3D_models/neck_pillow/3d_decimated/neck_pillow_2k.obj')
#bpy.ops.import_scene.obj(filepath='/home/arvardaz/SFT_with_CNN/3D_models/big_ball/3d_decimated/big_ball_2k.obj')
#name = 'ball'
name = 'pillow'



#Find object by name
obj_name = [v for v in bpy.data.objects.keys() if name in v][0]
obj = bpy.data.objects[obj_name]

bpy.ops.object.select_all(action='DESELECT')
obj.select = True
bpy.ops.xps_tools.convert_to_cycles_selected()

#Center the origin of object
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
obj.location = mathutils.Vector((0,0,0))

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
#    


look_at(bpy.data.objects['Camera'], mathutils.Vector((0,0,0)))


##
#set lamp brightness
#bpy.data.lamps['Lamp'].energy = 15
#bpy.data.lamps['new_lamp'].energy = 15
# SET RENDER TYPE
#bpy.context.scene.render.engine = 'CYCLES'

###
#import os
#imgName = os.path.expanduser('/home/arvardaz/SFT_with_CNN/3D_models/neck_pillow/3d_decimated/neck_pillow_2k.png')
#img = bpy.data.images.load(imgName)
#tex = bpy.data.textures.new('TexName', type = 'BLEND')
#tex.type = 'IMAGE' 
#print("Recast", tex, tex.type)
#tex = tex.type_recast()
#print("Done", tex, tex.type)
#tex.image = img
#mat = bpy.data.materials.new(name = 'MatName')
#mat.add_texture(texture = tex, texture_coordinates = 'ORCO', map_to = 'COLOR') 
#ob = bpy.context.object
#bpy.ops.object.material_slot_remove()
#me = ob.data
#me.add_material(mat)



###


#for i in np.arange(5):
    ## Assign random poses to object ###################################################
        
    #alpha = np.random.uniform(0, 360)
    #beta = np.random.uniform(0, 360)
    #gamma = np.random.uniform(0, 360)
    #obj.rotation_euler = mathutils.Euler((np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)))
    
    #std = 1
    #obj_x = std * np.random.randn()
    #obj_y = std * np.random.randn()
    #obj_z = std * np.random.randn()
    #obj.location = mathutils.Vector((obj_x, obj_y, obj_z))
    
##    # Assign random positions to camera and lamp ###################################################
##    cam_pos_min_x = -10
##    cam_pos_max_x = 10
##    cam_pos_min_y = -10
##    cam_pos_max_y = 10
##    cam_pos_min_z = -10
##    cam_pos_max_z = 10

##    new_cam_pos_x = np.random.uniform(cam_pos_min_x, cam_pos_max_x)
##    new_cam_pos_y = np.random.uniform(cam_pos_min_y, cam_pos_max_y)
##    new_cam_pos_z = np.random.uniform(cam_pos_min_z, cam_pos_max_z)

##    bpy.data.objects['Camera'].location = mathutils.Vector((new_cam_pos_x, new_cam_pos_y, new_cam_pos_z))

##    lamp_pos_min_x = -15
##    lamp_pos_max_x = 15
##    lamp_pos_min_y = -15
##    lamp_pos_max_y = 15
##    lamp_pos_min_z = -15
##    lamp_pos_max_z = 15
##    new_lamp_pos_x = np.random.uniform(lamp_pos_min_x, lamp_pos_max_x)
##    new_lamp_pos_y = np.random.uniform(lamp_pos_min_y, lamp_pos_max_y)
##    new_lamp_pos_z = np.random.uniform(lamp_pos_min_z, lamp_pos_max_z)
##    bpy.data.objects['Lamp'].location = mathutils.Vector((new_lamp_pos_x, new_lamp_pos_y, new_lamp_pos_z))

    ## Save Image #################################################################################################
    #bpy.context.scene.render.filepath = '/home/arvardaz/SFT_with_CNN/output/image_%02d.png' % i
    #bpy.context.scene.render.resolution_x = 400 
    #bpy.context.scene.render.resolution_y = 400
    #bpy.ops.render.render( write_still=True ) 
