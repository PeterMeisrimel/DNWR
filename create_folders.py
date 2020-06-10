#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:42:18 2020

@author: peter
"""

import numpy as np
import pylab as pl
import scipy as sp
pl.close('all')

import os

try: os.mkdir('verify') 
except: pass
try: os.mkdir('verify/base') 
except: pass
for i in ['DNWR', 'NNWR']:
    try: os.mkdir(f'verify/{i}') 
    except: pass
    for j in ['IE', 'SDIRK2', 'TA']:
        try: os.mkdir(f'verify/{i}/{j}')
        except: pass
        for k in ['air_steel', 'air_water', 'water_steel', 'extra_len']:
            try: os.mkdir(f'verify/{i}/{j}/{k}')
            except: pass            