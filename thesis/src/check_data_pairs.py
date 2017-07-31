#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:23:59 2017

@author: arvardaz
"""

from glob import glob1
import sys
import os

#path = '/home/arvardaz/SFT_with_CNN/datasets/train/'
#path = '/media/arvardaz/network_drive/SFT_with_CNN/datasets/dataset_rt+fl+l+bg_gpu/val/'

if __name__ == '__main__':
##    path = sys.argv[1]
#    csvs = glob1(path, '*.csv')
#    pngs = glob1(path, '*.png')
#    
#    csvs = [name[:-4] for name in csvs]
#    pngs = [name[:-4] for name in pngs]
#    
#    diffs = set(csvs).symmetric_difference(pngs)
#    print("differences: ",diffs)


    pathfrom = '/media/arvardaz/network_drive/SFT_with_CNN/datasets/dataset_rt+fl+l+bg_gpu/test/'
    pathto = '/media/arvardaz/network_drive/SFT_with_CNN/datasets/dataset_rt+fl+l+bg_gpu/val/'
    csvs = glob1(pathfrom, '*.csv')
    csvs = [name[:-4] for name in csvs]
    nlist = csvs[:1316]
    
    [os.rename(pathfrom+f+'.png', pathto+f+'.png') for f in nlist]
    [os.rename(pathfrom+f+'.csv', pathto+f+'.csv') for f in nlist]