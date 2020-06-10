#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:47:34 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np
import pylab as pl
import scipy as sp
from FSI_verification import get_problem, get_solver, get_parameters, get_init_cond
import json
from Problem_FSI_1D import Problem_FSI_1D
pl.close('all')

pl.rcParams['lines.linewidth'] = 2
pl.rcParams['font.size'] = 18
pl.rcParams['lines.markersize'] = 9

## measuring approximate convergence rate via the average update in the (up to) last 5 iterations
def get_conv_rates(tf, s = 10, n = 199, C1 = 1, C2 = 10, kmax = 6, **kwargs):
    init_cond = get_init_cond(1)
    
    results = {}
    
    prob = get_problem(dim = 1, n = n, **kwargs)
    solver = get_solver(prob, order = 1, WR_type = 'DNWR')
    nn = np.array([2**i for i in range(s)])
    
    results['para']  = [kwargs['alpha_1'], kwargs['alpha_2'], kwargs['lambda_1'], kwargs['lambda_2']]
    results['n'] = n
    results['tf'] = tf
    results['cfl'] = []
    
    theta_min = []
    theta_max = []
    theta_actual = []
    theta_avg = []
    
    dx = 1/(n+1)
    for n in nn:
        dt = tf/n
        results['cfl'].append(dt/(dx**2))
        dt1, dt2 = dt/C1, dt/C2
        th_min = prob.DNWR_theta_opt_test(min(dt1, dt2), min(dt1, dt2))
        
        _, _, _, updates, _ = solver(dt, C1, C2, init_cond, th_min, TOL = 1e-13, maxiter = kmax)
        theta_min.append(updates)
        
        th_max = prob.DNWR_theta_opt_test(max(dt1, dt2), max(dt1, dt2))
        _, _, _, updates, _ = solver(dt, C1, C2, init_cond, th_max, TOL = 1e-13, maxiter = kmax)
        theta_max.append(updates)
        
        th_act = prob.DNWR_theta_opt_test(dt1, dt2)
        _, _, _, updates, _ = solver(dt, C1, C2, init_cond, th_act, TOL = 1e-13, maxiter = kmax)
        theta_actual.append(updates)
        
        th_avg = prob.DNWR_theta_opt_test((dt1 + dt2)/2, (dt1 + dt2)/2)
        _, _, _, updates, _ = solver(dt, C1, C2, init_cond, th_avg, TOL = 1e-13, maxiter = kmax)
        theta_avg.append(updates)
    
    results['min'] = theta_min
    results['max'] = theta_max
    results['actual'] = theta_actual
    results['avg'] = theta_avg
    
    results['parameters'] = kwargs
    return results

def run_all(output_file, s = 10, n = 199, **kwargs):
    ## s = sampling rate
    # convergence rate for various theta parameters
    results = get_conv_rates(s = s, n = n, **kwargs)
    
    with open(output_file, 'w') as myfile:
        myfile.write(json.dumps(results, indent = 2, sort_keys = True))
        
def plotting(input_file, savefile):
    with open(input_file, 'r') as myfile:
        results = json.load(myfile)
    
    pl.figure()
    for up, label, marker in [(results['min'], 'Min', 'o'), (results['max'], 'Max', '*'), 
                               (results['actual'], 'Mix', '^'), (results['avg'], 'Avg', '+')]:
        conv_rates = []
        for updates in up:
            x = np.array(updates[:-1])
            conv_rates.append(np.mean(x[1:]/x[:-1]))
        pl.loglog(results['cfl'], conv_rates, label = label, marker = marker)
        
    pl.xlabel('C', labelpad = -20, position = (1.08, -1), fontsize = 20)
    lp = -50 if label == 'Air-Steel' else -70
    pl.ylabel('Conv. rate', rotation = 0, labelpad = lp, position = (2., 1.05), fontsize = 20)
    pl.ylim(1e-7, 2) ## manually setting limits
    pl.grid(True, 'major')
    pl.legend()
    pl.savefig(savefile, dpi = 100)
    
if __name__ == "__main__":
    kmax = 6
    for C1, C2 in [(1, 10), (10, 1)]:
        for tf, which in [(1e6, 'air_water'), (1e4, 'air_steel'), (1e4, 'water_steel')]:
            tf = int(tf)
            file = f'plots_data/MR_theta_opt_test_{which}_{C1}_{C2}_{tf}.txt'
#            run_all(file, **get_parameters(which), tf = tf, s = 40, n = 199, C1 = C1, C2 = C2)
            plotting(file, f'plots/MR_theta_opt_test_{which}_{C1}_{C2}_{tf}.png')