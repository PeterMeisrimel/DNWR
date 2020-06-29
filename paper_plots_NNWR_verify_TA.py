#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:03:21 2020

@author: Peter Meisrimel, Lund University
"""

import pylab as pl
from FSI_verification import get_problem, get_solver, get_parameters, get_init_cond
import json
pl.close('all')

pl.rcParams['lines.linewidth'] = 3
pl.rcParams['font.size'] = 14
pl.rcParams['lines.markersize'] = 14

def verify_adaptive(init_cond, tf = 1, k = 10, WR_type = 'NNWR', order = -1, **kwargs):
    ## sum of splitting and time-integration order
    ## calculating time-integration error( + splitting error) for a small tolerance
    if WR_type == 'NNWR':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
        ID_SELF = comm.rank
        
    prob = get_problem(WR_type = WR_type, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    if ID_SELF == 0: ## problem object for correct error computation
        prob_norm = get_problem(WR_type = 'MONOLITHIC', **kwargs)
    
    tols = [10**(-i) for i in range(k + 1)]
    
    if ID_SELF == 1:
        for tol in tols:
            solver(tf, init_cond, TOL = tol)
        return None # process 1 quits
    
    errs = {'u': [], 'it': [], 'timesteps': []}
    sols = []
    for tau in tols:
        print(tau)
        u1, u2, ug, E, it, s = solver(tf, init_cond, TOL = tau)
        sols.append((u1, u2, ug))
        errs['it'].append(it)
        errs['timesteps'].append(s)
    errs['tols'] = tols[:-1]
    
    u1_ref, u2_ref, ug_ref = sols[-1]
    for u in sols[:-1]:
        u1, u2, ug = u
        errs['u'].append(prob_norm.norm_inner(u1_ref - u1, u2_ref - u2, ug_ref - ug))
    del errs['it'][-1]; del errs['timesteps'][-1]
    errs['tf'] = tf
    
    return errs

def run_combined(k = 8, n = 32, dim = 1, savefile = None, **kwargs):
    init_cond = get_init_cond(dim)
    print('air steel')
    res_air_steel = verify_adaptive(init_cond, 1e4, k, n = n, dim = dim, **get_parameters('air_steel'))
    print('air water')
    res_air_water = verify_adaptive(init_cond, 1e4, k, n = n, dim = dim, **get_parameters('air_water'))
    print('water steel')
    res_water_steel = verify_adaptive(init_cond, 1e4, k, n = n, dim = dim, **get_parameters('water_steel'))
    
    if res_air_steel is not None:
        results = {'air_steel': res_air_steel, 'air_water': res_air_water, 'water_steel': res_water_steel,
                   'n': n, 'dim': dim}
        with open(savefile, 'w') as myfile:
            myfile.write(json.dumps(results, indent = 2, sort_keys = True))
        
def plotting(input_file, savefile):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.rank == 1: return None
    
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
    # mpiexec -n 2 python3 paper_plots_NNWR_verify_TA.py
    save_file_data_1 = 'plots_data/NNWR_TA_dim_1.txt'
    run_combined(k = 5, n = 199, dim = 1, savefile = save_file_data_1)
    plotting(save_file_data_1, 'plots/NNWR_TA_dim_1.png')
    
    save_file_data_2 = 'plots_data/NNWR_TA_dim_2.txt'
    run_combined(k = 4, n = 99, dim = 2, savefile = save_file_data_2)
    plotting(save_file_data_2, 'plots/NNWR_TA_dim_2.png')