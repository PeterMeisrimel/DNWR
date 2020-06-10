#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:11:05 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np
import pylab as pl
from FSI_verification import get_problem, get_solver, get_parameters, get_init_cond, solve_monolithic
import json
from mpi4py import MPI
pl.close('all')

pl.rcParams['lines.linewidth'] = 3
pl.rcParams['font.size'] = 18
pl.rcParams['lines.markersize'] = 12

## mpiexec -n 2 python3 paper_verify_orders_NNWR.py

def verify_comb_error(init_cond, tf = 1, k = 10, kmin = 0, C1 = 1, C2 = 1, theta = None, WR_type = 'NNWR', order = 1, TOL = 1e-13, **kwargs):
    comm = MPI.COMM_WORLD
    if comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
    ID_SELF = comm.rank
    ## sum of splitting and time-integration order
    ## calculating time-integration error( + splitting error) for a small tolerance
    prob = get_problem(WR_type = WR_type, **kwargs, )
    solver = get_solver(prob, order, WR_type)
    
    if ID_SELF == 0:
        u1_ref, u2_ref, ug_ref, _ = solve_monolithic(tf, int(max(C1, C2))*2**(k+1), init_cond, order, **kwargs)
        uref = np.hstack([u1_ref, u2_ref, ug_ref])
    elif ID_SELF == 1:
        pass
    
    n, dim = kwargs['n'], kwargs['dim']
    if dim == 1:  L2_fac = np.sqrt(1/(n + 1))
    else:         L2_fac = 1/(n + 1)
    
    if ID_SELF == 0:
        errs = {'u': [], 'it': [], 'dts': []}
        for n_steps in [2**i for i in range(kmin, k)]:
            errs['dts'].append(tf/n_steps)
            u1, u2, ug, E, it = solver(tf, C1*n_steps, C2*n_steps, init_cond, theta, TOL = TOL)
            errs['u'].append(np.linalg.norm(uref - np.hstack([u1, u2, ug]), 2)*L2_fac)
            errs['it'].append(it)
    elif ID_SELF == 1:
        for n_steps in [2**i for i in range(kmin, k)]:
            solver(tf, C1*n_steps, C2*n_steps, init_cond, theta, TOL = TOL)
        return None
    return errs

def run_combined(tf = 1, k1 = 10, k2 = 10, TOL = 1e-6, n1 = 100, n2 = 50, C1 = 1, C2 = 1, savefile = None, **kwargs):
    res_1d_IE = verify_comb_error(get_init_cond(1), tf, k1, C1 = C1, C2 = C2, order = 1, TOL = TOL, dim = 1, n = n1, **kwargs)
    res_1d_SD2 = verify_comb_error(get_init_cond(1), tf, k1, C1 = C1, C2 = C2, order = 2, TOL = TOL, dim = 1, n = n1, **kwargs)
    
    res_2d_IE = verify_comb_error(get_init_cond(2), tf, k2, C1 = C1, C2 = C2, order = 1, TOL = TOL, dim = 2, n = n2, **kwargs)
    res_2d_SD2 = verify_comb_error(get_init_cond(2), tf, k2, C1 = C1, C2 = C2, order = 2, TOL = TOL, dim = 2, n = n2, **kwargs)
    
    results = {'IE_1D': res_1d_IE, 'S2_1D': res_1d_SD2, 'IE_2D': res_2d_IE, 'S2_2D': res_2d_SD2,
               'tf': tf, 'n_1D': n1, 'n_2d': n2, 'C1': C1, 'C2': C2}
    with open(savefile, 'w') as myfile:
        myfile.write(json.dumps(results, indent = 2, sort_keys = True))
        
def plot_combined(input_file, savefile, hide = False):
    comm = MPI.COMM_WORLD
    if comm.rank == 1: return None
    
    with open(input_file, 'r') as myfile:
        results = json.load(myfile)
    res_IE_1D = results['IE_1D']
    res_IE_2D = results['IE_2D']
    res_S2_1D = results['S2_1D']
    res_S2_2D = results['S2_2D']

    pl.figure()
    pl.loglog(res_IE_1D['dts'], res_IE_1D['u'], label = 'IE, 1D', marker = 'o')
    if not hide:
        pl.loglog(res_IE_2D['dts'], res_IE_2D['u'], label = 'IE, 2D', marker = 'd')
    pl.loglog(res_S2_1D['dts'], res_S2_1D['u'], label = 'SD2, 1D', marker = '+')
    if not hide: 
        pl.loglog(res_S2_2D['dts'], res_S2_2D['u'], label = 'SD2, 2D', marker = 'v')
    dts = np.array(res_IE_1D['dts'])
    pl.loglog(dts, dts/dts[0]*res_IE_1D['u'][0]*10, label = '1st', linestyle = ':')
    pl.loglog(res_IE_1D['dts'], (dts/dts[0])**2*res_S2_1D['u'][0]*10, label = '2nd', linestyle = '--')
    pl.xlabel(r'$\Delta t$', labelpad = -20, position = (1.05, -1), fontsize = 20)
    pl.ylabel('Err', rotation = 0, labelpad = -50, position = (2., 1.0), fontsize = 20)
    pl.grid(True, 'major')
#    pl.title(r'Error over $\Delta t$')
    pl.legend()
    pl.savefig(savefile, dpi = 100)
    
    pl.figure()
    pl.semilogx(res_IE_1D['dts'], res_IE_1D['it'], label = 'IE, 1D', marker = 'o')
    pl.semilogx(res_IE_2D['dts'], res_IE_2D['it'], label = 'IE, 2D', marker = 'd')
    pl.semilogx(res_S2_1D['dts'], res_S2_1D['it'], label = 'SD2, 1D', marker = '+')
    pl.semilogx(res_S2_2D['dts'], res_S2_2D['it'], label = 'SD2, 2D', marker = 'v')
    dts = np.array(res_IE_1D['dts'])
    pl.xlabel(r'$\Delta t$', labelpad = -20, position = (1.05, -1), fontsize = 20)
    pl.ylabel('Iters', rotation = 0, labelpad = -50, position = (2., 1.0), fontsize = 20)
    pl.grid(True, 'major')
#    pl.title(r'Error over $\Delta t$')
    pl.legend()
    pl.savefig(savefile[:-4] + '_iters.png', dpi = 100)
    
if __name__ == '__main__':
    # mpiexec -n 2 python3 paper_plots_NNWR_verify_orders.py
    for tf, which, C1, C2 in [(1e6, 'air_water', 1, 1), (1e4, 'air_steel', 1, 10), (1e4, 'water_steel', 10, 1)]:
        add = ''
        save_file_data = f'plots_data/NNWR_{which}_{C1}_{C2}_MR_order{add}_new.txt'
        run_combined(tf = tf, k1 = 6, k2 = 5, n1 = 199, n2 = 99, C1 = C1, C2 = C2, savefile = save_file_data, **get_parameters(which))
        plot_combined(save_file_data, f'plots/NNWR_{which}_{C1}_{C2}_MR_order{add}_new.png', which == 'water_steel')