#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:55:21 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np
import pylab as pl
import scipy as sp
from FSI_verification import get_problem, get_solver, get_parameters, get_init_cond, solve_monolithic
import json
pl.close('all')

pl.rcParams['lines.linewidth'] = 3
pl.rcParams['font.size'] = 14
pl.rcParams['lines.markersize'] = 14

def verify_adaptive(init_cond, tf = 1, k = 10, WR_type = 'DNWR', order = -2, **kwargs):
    ## sum of splitting and time-integration order
    ## calculating time-integration error( + splitting error) for a small tolerance
    prob = get_problem(WR_type = WR_type, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    n, dim = kwargs['n'], kwargs['dim']
    if dim == 1:  L2_fac = np.sqrt(1/(n + 1))
    else:         L2_fac = 1/(n + 1)
    
    errs = {'u': [], 'it': [], 'timesteps': []}
    sols = []
    tols = [10**(-i) for i in range(k + 1)]
    for tau in tols:
        u1, u2, ug, E, it, s = solver(tf, init_cond, TOL = tau)
        sols.append(np.hstack([u1, u2, ug]))
        errs['it'].append(it)
        errs['timesteps'].append(s)
    errs['tols'] = tols[:-1]
    
    uref = sols[-1]
    for u in sols[:-1]:
        errs['u'].append(np.linalg.norm(u - uref, 2)*L2_fac)
    del errs['it'][-1]; del errs['timesteps'][-1]
    errs['tf'] = tf
    errs['dim'] = dim
    
    return errs

def run_combined(k = 8, n = 32, dim = 1, savefile = None, **kwargs):
    init_cond = get_init_cond(dim)
    res_air_steel = verify_adaptive(init_cond, 1e4, k, n = n, dim = dim, **get_parameters('air_steel'))
    res_air_water = verify_adaptive(init_cond, 1e6, k, n = n, dim = dim, **get_parameters('air_water'))
    res_water_steel = verify_adaptive(init_cond, 1e4, k, n = n, dim = dim, **get_parameters('water_steel'))
    
    results = {'air_steel': res_air_steel, 'air_water': res_air_water, 'water_steel': res_water_steel,
               'n': n, 'dim': dim}
    with open(savefile, 'w') as myfile:
        myfile.write(json.dumps(results, indent = 2, sort_keys = True))
        
def plotting(input_file, savefile):
    with open(input_file, 'r') as myfile:
        results = json.load(myfile)
    air_steel = results['air_steel']
    air_water = results['air_water']
    water_steel = results['water_steel']

    pl.figure()
    pl.loglog(air_water['tols'], air_water['u'], label = 'Air-Water', marker = 'd')
    pl.loglog(air_steel['tols'], air_steel['u'], label = 'Air-Steel', marker = 'o')
    pl.loglog(water_steel['tols'], water_steel['u'], label = 'Water-Steel', marker = '+')
    
    pl.loglog(air_steel['tols'], air_steel['tols'], label = 'TOL', color = 'k', linestyle = '--')
    pl.xlabel('TOL')
    pl.ylabel('Err')
    pl.grid(True, 'major')
#    pl.title('Error in time adpative case, dim = {}'.format(air_steel['dim']))
#    pl.title('Error in time adpative case')
    pl.legend()
    pl.savefig(savefile, dpi = 100)
    
if __name__ == '__main__':
    # running this took wayyy too long
    save_file_data_1 = f'plots_data/TA_dim_1.txt'
#    run_combined(k = 8, n = 199, dim = 1, savefile = save_file_data_1)
    plotting(save_file_data_1, 'plots/TA_dim_1.png')
    
    save_file_data_2 = f'plots_data/TA_dim_2.txt'
#    run_combined(k = 6, n = 99, dim = 2, savefile = save_file_data_2)
    plotting(save_file_data_2, 'plots/TA_dim_2.png')