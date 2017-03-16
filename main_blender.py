import bpy
import os

filename = os.path.join(os.path.basename(bpy.data.filepath), "/home/arvardaz/SFT_with_CNN/myblend_norm.py")
exec(compile(open(filename).read(), filename, 'exec'))