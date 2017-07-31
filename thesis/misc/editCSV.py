#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:23:02 2017

@author: arvardaz
"""

import re

from glob import glob1

#datapath = "../datasets/dataset_rt+fl/"
datapath = "/home/arvardaz/SFT_with_CNN/3D_models/tube/extra/3d/2k/dataset/"

filelist = glob1(datapath, "*.csv")
for fname in filelist[2:]:
    f = open(datapath+fname, 'r+')
    text = f.read()
#    text = re.sub(r"^(.+/)(.+.png,)", r"\2", text)
    text = re.sub(r"(^.+.+png,\d+\.\d+)", r"\1,1.0", text)
    f.seek(0)
    f.write(text)
    f.truncate()
    f.close()