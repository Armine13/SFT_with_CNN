#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:23:02 2017

@author: arvardaz
"""

import re

from glob import glob1

datapath = "../datasets/dataset_rt+fl/"


filelist = glob1(datapath, "*.csv")
for fname in filelist:
    f = open(datapath+fname, 'r+')
    text = f.read()
    text = re.sub(r"^(.+/)(.+.png,)", r"\2", text)
    f.seek(0)
    f.write(text)
    f.truncate()
    f.close()