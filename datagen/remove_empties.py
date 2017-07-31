#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:24:35 2017

@author: arvardaz
"""
#%%
import numpy as np
from glob import glob1
from shutil import copyfile, move
from os import remove

#%%
dir = '/home/arvardaz/Dropbox/datasets/paper/train/'
#dir='/home/isit/armine/Dropbox/tube/train/'
outdir = '/home/arvardaz/Dropbox/datasets/paper_empties/train/'
csv_list = glob1(dir,'*.csv')
#csv_list_path = [dir+s for s in csv_list]

for file_path in csv_list:
    data_str = np.genfromtxt(dir+file_path, delimiter=',', dtype=str)
    
#    if data_str.size == 0:
#        remove(dir+file_path)
#        remove(dir+file_path[:-3]+'png')
#        continue
        
    im_name = data_str[0]
    #fl = float(data_str[1])
    presence = int(data_str[2])
    if presence == 0:
        move(dir+file_path, outdir+file_path)
        move(dir+im_name, outdir+im_name)
#        copyfile(dir+file_path, outdir+file_path)
#        copyfile(dir+im_name, outdir+im_name)
#        remove(dir+file_path)
#        remove(dir+im_name)
        

    