#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:01:39 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np
import pylab as pl
import scipy as sp
pl.close('all')

from Problem_FSI_1D import Problem_FSI_1D
from FSI_verification import get_parameters

pl.rcParams['lines.linewidth'] = 3
pl.rcParams['font.size'] = 16
pl.rcParams['ytick.labelsize'] = 14

def get_opt_theta(n, T, steps_range, which, label = 'label'):
    parameters = get_parameters(which)
    prob = Problem_FSI_1D(n = n, **parameters)
    dx = prob.dx
    
    N_steps = np.array([int(2**i) for i in np.linspace(0, 50, 1000)])
    dts = T/N_steps
    
    CFL = [dt/(dx**2) for dt in dts]
    theta = [prob.DNWR_theta_opt(dt, dt) for dt in dts]
    lim_zero = parameters['alpha_2']/(parameters['alpha_1'] + parameters['alpha_2'])
    lim_inf = parameters['lambda_2']/(parameters['lambda_1'] + parameters['lambda_2'])
    
    pl.figure()
    pl.semilogx(CFL, theta, label = label)
    pl.axhline(lim_zero, ls = '--', color = 'k', label = r'$\theta_{c \rightarrow 0}$')
    pl.axhline(lim_inf, ls = ':', color = 'k', label = r'$\theta_{c \rightarrow \infty}$')
    pl.legend()
#    pl.xlabel(r'$c = \frac{\Delta t}{\Delta x^2}$', labelpad = -30, position = (1.05, -1), fontsize = 20)
    pl.xlabel('c', labelpad = -30, position = (1.05, -1), fontsize = 20)
    lp = -50 if label == 'Air-Steel' else -20
    pl.ylabel(r'$\theta_{opt}$', rotation = 0, labelpad = lp, position = (2., 1.), fontsize = 20)
    pl.title(label)
    pl.savefig('plots/CFL_vs_opt_theta_{}.png'.format(which), dpi = 100)
    
    return 

n = 99
T = 1e10
n_steps = 50
get_opt_theta(n, T, n_steps, 'air_water', 'Air-Water')
get_opt_theta(n, T, n_steps, 'air_steel', 'Air-Steel')
get_opt_theta(n, T, n_steps, 'water_steel', 'Water-Steel')