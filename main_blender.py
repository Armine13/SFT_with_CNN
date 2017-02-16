import bpy
import os

import imp
foo = imp.load_source('saveMeshAsCloud', '/home/arvardaz/SFT_with_CNN/saveMeshAsCloud.py')

#filename = os.path.join(os.path.basename(bpy.data.filepath), "/home/arvardaz/SFT_with_CNN/myblend.py")
#exec(compile(open(filename).read(), filename, 'exec'))

#for vertex in item.data.vertices:
#   print(vertex.index, vertex.co)