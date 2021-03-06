#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:55:21 2020

@author: Peter Meisrimel, Lund University
"""

import pylab as pl
from FSI_verification import get_problem, get_solver, get_parameters, get_init_cond
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
    
    errs = {'u': [], 'it': [], 'timesteps': []}
    sols = []
    tols = [10**(-i) for i in range(k + 1)]
    for tau in tols:
        u1, u2, ug, E, it, s = solver(tf, init_cond, TOL = tau)
        sols.append((u1, u2, ug))
        errs['it'].append(it)
        errs['timesteps'].append(s)
    errs['tols'] = tols[:-1]
    
    u1_ref, u2_ref, ug_ref = sols[-1]
    for u in sols[:-1]:
        u1, u2, ug = u
        errs['u'].append(prob.norm_inner(u1_ref - u1, u2_ref - u2, ug_ref - ug))
    del errs['it'][-1]; del errs['timesteps'][-1]
    errs['tf'] = tf
    
    return errs

def run_combined(k = 8, n = 32, dim = 1, savefile = None, **kwargs):
    init_cond = get_init_cond(dim)
    res_air_steel = verify_adaptive(init_cond, 1e4, k, n = n, dim = dim, **get_parameters('air_steel'))
    res_air_water = verify_adaptive(init_cond, 1e4, k, n = n, dim = dim, **get_parameters('air_water'))
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
    pl.legend()
    pl.savefig(savefile, dpi = 100)
    
if __name__ == '__main__':
    save_file_data_1 = 'plots_data/TA_dim_1.txt'
    run_combined(k = 8, n = 199, dim = 1, savefile = save_file_data_1)
    plotting(save_file_data_1, 'plots/TA_dim_1.png')
    
    save_file_data_2 = 'plots_data/TA_dim_2.txt'
    run_combined(k = 7, n = 99, dim = 2, savefile = save_file_data_2)
    plotting(save_file_data_2, 'plots/TA_dim_2.png')