import os
import sys
from PIL import Image

def resize(folder, fileName, outPath, size):
    filePath = os.path.join(folder, fileName)
    im = Image.open(filePath)
    w, h  = im.size
    newIm = im.resize((size[0], size[1]))
    # i am saving a copy, you can overrider orginal, or save to other folder
    newIm.save(outPath+fileName+"100x100.png")

def bulkResize(imageFolder, outPath, size):
    imgExts = ["png", "bmp", "jpg"]
    for path, dirs, files in os.walk(imageFolder):
        for fileName in files:
            ext = fileName[-3:].lower()
            if ext not in imgExts:
                continue

            resize(path, fileName, outPath, size)

if __name__ == "__main__":
    imageFolder='./output/' # first arg is path to image folder
    outFolder = './output100x100/'
    newsize = (224, 224)

    if not os.path.exists(os.path.dirname(outFolder)):
        try:
            os.makedirs(os.path.dirname(outFolder))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    
    bulkResize(imageFolder, outFolder, newsize)
