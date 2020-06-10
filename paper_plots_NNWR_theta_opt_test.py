#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:47:34 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np
import pylab as pl
from FSI_verification import get_problem, get_solver, get_parameters, get_init_cond
import json
from Problem_FSI_1D import Problem_FSI_1D
from mpi4py import MPI
pl.close('all')

pl.rcParams['lines.linewidth'] = 2
pl.rcParams['font.size'] = 18
pl.rcParams['lines.markersize'] = 9

## mpiexec -n 2 python3 paper_theta_opt_test_NNWR.py

## measuring approximate convergence rate via the average update in the (up to) last 5 iterations
def get_conv_rates(init_cond, tf, N_steps = 100, s = 100, order = 1, dim = 1, n = 32, C1 = 1, C2 = 2, kmax = 6, thmin = 0, thmax = 1, **kwargs):
    comm = MPI.COMM_WORLD
    if comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
    ID_SELF = comm.rank
    print(order, dim, kwargs)
    
    CFL_zero_lim = kwargs['alpha_2']*kwargs['alpha_1']/(kwargs['alpha_1'] + kwargs['alpha_2'])**2
    CFL_inf_lim = kwargs['lambda_2']*kwargs['lambda_1']/(kwargs['lambda_1'] + kwargs['lambda_2'])**2
    
    results = {'updates': [], 'theta': list(np.linspace(thmin, thmax, s))}
    
    prob = get_problem(dim = dim, n = n, WR_type = 'NNWR', **kwargs)
    solver = get_solver(prob, order = order, WR_type = 'NNWR')
    
    if ID_SELF == 0:
        for th in results['theta']:
            print('theta', th)
            _, _, _, updates, _ = solver(tf, C1*N_steps, C2*N_steps, init_cond, th, TOL = 1e-13, maxiter = kmax)
            results['updates'].append(updates)
    elif ID_SELF == 1:
        for th in results['theta']:
            solver(tf, C1*N_steps, C2*N_steps, init_cond, th, TOL = 1e-13, maxiter = kmax)
        return None ## process 2 quitting
        
    ## calculate theta_opt for given setting
    dt1, dt2 = tf/(N_steps*C1), tf/(N_steps*C2)
    results['theta_opt'] = prob.NNWR_theta_opt(dt1, dt2)
    results['theta_CFL_zero'] = CFL_zero_lim
    results['theta_CFL_inf'] = CFL_inf_lim
    results['theta_start'] = thmin
    results['theta_stop'] = thmax
    
    results['parameters'] = kwargs
    results['tf'] = tf
    results['n'] = n
    results['N_steps'] = N_steps
    return results

def run_all(output_file, s, N_steps = 100, n1 = 50, n2 = 32, thmin = 0, thmax = 1, **kwargs):
#def run_all(output_file, s, N_steps = 1, n1 = 50, n2 = 32, **kwargs):
    ## s = sampling rate
    # convergence rate for various theta parameters
    res_IE_1D = get_conv_rates(N_steps = N_steps, s = s, order = 1, dim = 1, n = n1, init_cond = get_init_cond(1), thmin = thmin, thmax = thmax, **kwargs)
    res_IE_2D = get_conv_rates(N_steps = N_steps, s = s, order = 1, dim = 2, n = n2, init_cond = get_init_cond(2), thmin = thmin, thmax = thmax, **kwargs)
    res_S2_1D = get_conv_rates(N_steps = N_steps, s = s, order = 2, dim = 1, n = n1, init_cond = get_init_cond(1), thmin = thmin, thmax = thmax, **kwargs)
    res_S2_2D = get_conv_rates(N_steps = N_steps, s = s, order = 2, dim = 2, n = n2, init_cond = get_init_cond(2), thmin = thmin, thmax = thmax, **kwargs)
    
    results = {'IE_1D': res_IE_1D, 'IE_2D': res_IE_2D, 'S2_1D': res_S2_1D, 'S2_2D': res_S2_2D}
    with open(output_file, 'w') as myfile:
        myfile.write(json.dumps(results, indent = 2, sort_keys = True))
        
def plotting(input_file, savefile):
    comm = MPI.COMM_WORLD
    if comm.rank == 1: return None
    
    with open(input_file, 'r') as myfile:
        results = json.load(myfile)
    res_IE_1D = results['IE_1D']
    
    thetas = np.linspace(res_IE_1D['theta_start'], res_IE_1D['theta_stop'], 1000)
    p = Problem_FSI_1D(res_IE_1D['n'], **res_IE_1D['parameters'], WR_type = 'NNWR')
    dt = res_IE_1D['tf']/res_IE_1D['N_steps']
    conv_rates_theo = [p.theoretical_conv_rate_NNWR(dt, th) for th in thetas]

    pl.figure()
    for res, label, marker, ls in [(results['IE_1D'], 'IE 1D', None, '-'), (results['IE_2D'], 'IE 2D', None, '-'), 
                               (results['S2_1D'], 'SD2 1D', 'o', ''), (results['S2_2D'], 'SD2 2D', '*', '')]:
        conv_rates = []
        for updates in res['updates']:
            x = np.array(updates[:-1])
            conv_rates.append(np.mean(x[1:]/x[:-1]))
        pl.semilogy(res['theta'], conv_rates, label = label, marker = marker, linestyle = ls)
    
    pl.semilogy(thetas, conv_rates_theo, label = r'$\Sigma(\Theta)$', ls = '--', linewidth  = 3)
    pl.axvline(p.NNWR_theta_opt(dt, dt), ls = '-', color = 'k', label = r'$\Theta_{opt}$')
    pl.axhline(1, color = 'k', ls = '--', linewidth = 1)
    a, b = res_IE_1D['theta_CFL_inf'], res_IE_1D['theta_CFL_zero']
    pl.ylim(pl.ylim())
    pl.fill_between([min(a,b), max(a,b)], [min(pl.ylim())/100]*2, [max(pl.ylim())*100]*2, alpha = 0.2)
    pl.xlim(max(0,res_IE_1D['theta_start']), res_IE_1D['theta_stop'])
    pl.xlabel(r'$\Theta$', labelpad = -20, position = (1.11, -1), fontsize = 20)
    lp = -50 if label == 'Air-Steel' else -70
    pl.ylabel('Conv. rate', rotation = 0, labelpad = lp, position = (2., 1.05), fontsize = 20)
    pl.legend()
    pl.savefig(savefile, dpi = 100)
    
if __name__ == "__main__":
    ## mpiexec -n 2 python3 paper_plots_NNWR_theta_opt_test.py
    kmax = 6
    for tf, which, C1, C2, thmin, thmax in [(1e6, 'air_water', 1, 1, 0, 0.06),
                                            (1e4, 'air_steel', 1, 10, 0, 0.0008),
                                            (1e4, 'water_steel', 10, 1, 0, 0.3)]:
        tf = int(tf)
        file = f'plots_data/NNWR_theta_opt_test_{which}_{C1}_{C2}_{tf}.txt'
#        run_all(file, **get_parameters(which), tf = tf, n1 = 199, n2 = 99, s = 30, C1 = C1, C2 = C2, thmin = thmin, thmax = thmax)
        plotting(file, f'plots/NNWR_theta_opt_test_{which}_{C1}_{C2}_{tf}.png')