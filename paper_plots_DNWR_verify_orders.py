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
pl.close('all')

pl.rcParams['lines.linewidth'] = 3
pl.rcParams['font.size'] = 18
pl.rcParams['lines.markersize'] = 12

def verify_comb_error(init_cond, tf = 1, k = 10, kmin = 0, C1 = 1, C2 = 1, theta = None, WR_type = 'DNWR', order = 1, TOL = 1e-13, **kwargs):
    ## sum of splitting and time-integration order
    ## calculating time-integration error( + splitting error) for a small tolerance
    prob = get_problem(WR_type = WR_type, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    u1_ref, u2_ref, ug_ref, _, prob_mono = solve_monolithic(tf, int(max(C1, C2))*2**(k+1), init_cond, order, **kwargs)
    
    errs = {'u': [], 'it': [], 'dts': []}
    for n_steps in [2**i for i in range(kmin, k)]:
        errs['dts'].append(tf/n_steps)
        u1, u2, ug, E, it = solver(tf, C1*n_steps, C2*n_steps, init_cond, theta, TOL = TOL)
        errs['u'].append(prob_mono.norm_inner(u1_ref - u1, u2_ref - u2, ug_ref - ug))
        errs['it'].append(it)
    return errs

def run_combined(tf = 1, k1 = 10, k2 = 10, TOL = 1e-13, n1 = 100, n2 = 50, C1 = 1, C2 = 1, savefile = None, **kwargs):
    res_1d_IE = verify_comb_error(get_init_cond(1), tf, k1, C1 = C1, C2 = C2, order = 1, TOL = TOL, dim = 1, n = n1, **kwargs)
    res_1d_SD2 = verify_comb_error(get_init_cond(1), tf, k1, C1 = C1, C2 = C2, order = 2, TOL = TOL, dim = 1, n = n1, **kwargs)
    
    res_2d_IE = verify_comb_error(get_init_cond(2), tf, k2, C1 = C1, C2 = C2, order = 1, TOL = TOL, dim = 2, n = n2, **kwargs)
    res_2d_SD2 = verify_comb_error(get_init_cond(2), tf, k2, C1 = C1, C2 = C2, order = 2, TOL = TOL, dim = 2, n = n2, **kwargs)
    
    results = {'IE_1D': res_1d_IE, 'S2_1D': res_1d_SD2, 'IE_2D': res_2d_IE, 'S2_2D': res_2d_SD2,
               'tf': tf, 'n_1D': n1, 'n_2d': n2, 'C1': C1, 'C2': C2}
    with open(savefile, 'w') as myfile:
        myfile.write(json.dumps(results, indent = 2, sort_keys = True))
        
def plot_combined(input_file, savefile, which):
    with open(input_file, 'r') as myfile:
        results = json.load(myfile)
    res_IE_1D = results['IE_1D']
    res_IE_2D = results['IE_2D']
    res_S2_1D = results['S2_1D']
    res_S2_2D = results['S2_2D']

    pl.figure()
    pl.loglog(res_IE_1D['dts'], res_IE_1D['u'], label = 'IE, 1D', marker = 'o')
    pl.loglog(res_IE_2D['dts'], res_IE_2D['u'], label = 'IE, 2D', marker = 'd')
    pl.loglog(res_S2_1D['dts'], res_S2_1D['u'], label = 'SD2, 1D', marker = '+')
    pl.loglog(res_S2_2D['dts'], res_S2_2D['u'], label = 'SD2, 2D', marker = 'v')
    
    dts = np.array(res_IE_1D['dts'])
    pl.loglog(dts, dts/dts[0]*(res_IE_1D['u'][0] + res_IE_2D['u'][0])/10, label = '1st', linestyle = ':')
    pl.loglog(dts, (dts/dts[0])**2*(res_S2_1D['u'][0] + res_S2_2D['u'][0])/100, label = '2nd', linestyle = '--')
    pl.xlabel(r'$\Delta t$', labelpad = -20, position = (1.05, -1), fontsize = 20)
    pl.ylabel('Err', rotation = 0, labelpad = -50, position = (2., 1.03), fontsize = 20)
    pl.grid(True, 'major')
    pl.legend(loc = 2)
    pl.savefig(savefile, dpi = 100)
    
if __name__ == '__main__':
    for tf, which, C1, C2 in [(1, 'air_water', 10, 1),
                              (1, 'air_steel', 1, 1),
                              (1, 'water_steel', 1, 10)]:
        tf = int(tf)
        save_file_data = 'plots_data/DNWR_MR_order_{}_{}_{}_{}.txt'.format(which, C1, C2, tf)
#        run_combined(tf = tf, k1 = 6, k2 = 5, n1 = 9, n2 = 9, C1 = C1, C2 = C2, savefile = save_file_data, **get_parameters(which))
        run_combined(tf = tf, k1 = 7, k2 = 7, n1 = 199, n2 = 99, C1 = C1, C2 = C2, savefile = save_file_data, **get_parameters(which))
        plot_combined(save_file_data, 'plots/DNWR_MR_order_{}_{}_{}_{}.png'.format(which, C1, C2, tf), which)