#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:43:59 2020

@author: Peter Meisrimel, Lund University
"""

import pylab as pl
from FSI_verification import get_problem, get_solver, get_parameters, get_init_cond, solve_monolithic
import json

pl.rcParams['lines.linewidth'] = 2
pl.rcParams['font.size'] = 18
pl.rcParams['lines.markersize'] = 12

"""
Aim: Make an error work comparison between adaptive and non-adaptive

Problem #1: multirate needs dt, resp. N (N1, N2), and TOL as inputs.

Solution for how to choose N1 to N2 ratio: 
Run adaptive and denote the ratio of timesteps (1D & 2D separate?)
Use this result as basis for determining N1 and N2.

Solution for TOL: 
Compute time-integration error (!!!) for MR, using a very small tolerance.
use tol = err/5 as tolerance, now for both MR and adaptive case 
    
Suitable multirate timestep ratios as given by material parameters
"""      

## 2D only
def get_err_work(output_file, init_cond, tf, C1 = 1, C2 = 1, n = 99, k_mr = 6, k_adaptive = 6, **kwargs):
    results = kwargs.copy()
    results['C1'] = C1
    results['C2'] = C2
    results['tf'] = tf
    results['n'] = n
    
    ## step 1, calculate MR time-integration errors
    # a) get MR solutions with small tolerance
    prob = get_problem(dim = 2, n = n, WR_type = 'DNWR', **kwargs)
    solver_MR = get_solver(prob, order = 2, WR_type = 'DNWR')
    
    steps_list = [2**i for i in range(k_mr)]
    sols = []
    for s in steps_list:
        u1, u2, ug, _, _ = solver_MR(tf, s*C1, s*C2, init_cond, TOL = 1e-12)
        sols.append((u1, u2, ug))
    results['MR_base_sols'] = [(list(i[0]), list(i[1]), list(i[2])) for i in sols]
            
    ## b) get monolithic reference sol with even smaller dt
    u1ref, u2ref, ugref, _, prob_mono = solve_monolithic(tf, max(C1, C2)*2**k_mr, init_cond, 2, dim = 2, n = n, **kwargs)
    results['MR_ref'] = (list(u1ref), list(u2ref), list(ugref))
    
    ## c) calculate errors
    errs = []
    for u in sols:
        u1, u2, ug = u
        errs.append(prob_mono.norm_inner(u1  - u1ref, u2 - u2ref, ug - ugref))
    results['errs_for_tols'] = errs
    
    ## step 2, compute new, proper MR solutions
    sols_MR_new, steps_MR_new, iters_MR_new = [], [], []
    for s, e in zip(steps_list, errs):
        print(s, steps_list)
        tol = e/5
        u1, u2, ug, _, iters = solver_MR(tf, s*C1, s*C2, init_cond, TOL = tol)
        sols_MR_new.append((u1, u2, ug))
        steps_MR_new.append(iters*(s*C1 + s*C2))
        iters_MR_new.append(iters)
        
    results['sols_MR_new'] = [(list(u[0]), list(u[1]), list(u[2])) for u in sols_MR_new]
    results['steps_MR_new'] = steps_MR_new
    results['iters_MR_new'] = iters_MR_new
    
    ## step 3, compute adaptive solutions
    solver_adaptive = get_solver(prob, order = -2, WR_type = 'DNWR')
    sols_adaptive, steps_adaptive, iters_adaptive = [], [], []
    tols_list = [10**(-i) for i in range(k_adaptive)]
    for tol in tols_list:
        u1, u2, ug, _, iters, ss = solver_adaptive(tf, init_cond, TOL = tol)
        sols_adaptive.append((u1, u2, ug))
        steps_adaptive.append(ss)
        iters_adaptive.append(iters)
    
    results['sols_adaptive'] = [(list(u[0]), list(u[1]), list(u[2])) for u in sols_adaptive]
    results['steps_adaptive'] = steps_adaptive
    results['iters_adaptive'] = iters_adaptive
    
    ## step 4, reference solution for error computation, here: use adaptive for one tol further
    u1ref, u2ref, ugref, _, _, _ = solver_adaptive(tf, init_cond, TOL = 10**(-k_adaptive))
    results['err_ref'] = (list(u1ref), list(u2ref), list(ugref))
    
    ## step 5, compute errors
    errs_MR = []
    for u in sols_MR_new:
        u1, u2, ug = u
        errs_MR.append(prob_mono.norm_inner(u1  - u1ref, u2 - u2ref, ug - ugref))
        
    errs_adaptive = []
    for u in sols_adaptive:
        u1, u2, ug = u
        errs_adaptive.append(prob_mono.norm_inner(u1  - u1ref, u2 - u2ref, ug - ugref))
        
    results['errs_MR'] = errs_MR
    results['errs_adaptive'] = errs_adaptive
    
    with open(output_file, 'w') as myfile:
        myfile.write(json.dumps(results, indent = 2, sort_keys = True))
        
def plotting(input_file, savefile):
    with open(input_file, 'r') as myfile:
        results = json.load(myfile)
    
    pl.figure()
    pl.loglog(results['steps_MR_new'], results['errs_MR'], label = 'multirate', linestyle = '--', marker = 'o')
    pl.loglog(results['steps_adaptive'], results['errs_adaptive'], label = 'adaptive', linestyle = '-', marker = '*')
    pl.xlabel('Work', labelpad = -20, position = (1.05, -1), fontsize = 20)
    pl.ylabel('Err', rotation = 0, labelpad = -50, position = (2., 1.0), fontsize = 20)
    pl.xlim(1, 5e5)
    pl.legend()
    pl.grid(True, 'major')
    pl.savefig(savefile, dpi = 100)
    
    pl.figure()
    pl.plot(results['iters_MR_new'], label = 'multirate', linestyle = '--', marker = 'o')
    pl.plot(results['iters_adaptive'], label = 'adaptive', linestyle = '-', marker = '*')
    pl.ylabel('Iters', rotation = 0, labelpad = -30, position = (2., 1.0), fontsize = 20)
    pl.legend()
    pl.savefig(savefile[:-4] + '_iters.png', dpi = 100)

if __name__ == '__main__':
    n, k_mr, k_adap = 99, 6, 6
    for i in [1, 2]:
#        for tf, which in [(1e4, 'air_water')]:
#        for tf, which in [(1e4, 'air_steel')]:
        for tf, which in [(1e4, 'water_steel')]:
            para = get_parameters(which)
            D1 = para['lambda_1']/para['alpha_1']
            D2 = para['lambda_2']/para['alpha_2']
            
            C1 = max(1, int(D1/D2))
            C2 = max(1, int(D2/D1))
            print(i, tf, which, C1, C2)
            out_file = 'plots_data/err_work_u0_{}_{}.txt'.format(i, which)
            get_err_work(out_file, get_init_cond(2, num = i), tf, C1, C2, n, k_mr, k_adap, **get_parameters(which))
            plotting(out_file, 'plots/err_work_u0_{}_{}.png'.format(i, which))