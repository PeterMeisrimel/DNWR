#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:20:34 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np

#############################################################
#  SDIRK 2, Multirate
#############################################################    
def DNWR_SDIRK2(self, tf, N1, N2, init_cond, theta = None, maxiter = 100, TOL = 1e-8):
    if self.WR_type != 'DNWR': raise ValueError('invalid solution method')
    dt1, dt2 = tf/N1, tf/N2
    if theta is None: theta = self.DNWR_theta_opt(dt1, dt2)
    
    u10, u20, ug0 = self.get_initial_values(init_cond) ## different for 1D and 2D
    flux0 = np.zeros(ug0.shape)
    
    t1, t2 = np.linspace(0., tf, N1+1), np.linspace(0., tf, N2+1) # time grids
    ug_WF_old = [np.copy(ug0) for i in range(N2 + 1)] # waveform of ug, N grid
    ug_WF_new = [np.copy(ug0) for i in range(N2 + 1)] # waveform of ug, N grid
        
    t1_stage = [0.] + [i*dt1 + self.a*dt1 for i in range(N1)] # time grid for flux stage values
    flux_WF_1 = [np.copy(flux0) for i in range(N1 + 1)] # waveform of fluxes, D grid, first stage
    flux_WF_2 = [np.copy(flux0) for i in range(N1 + 1)] # waveform of fluxes, D grid, second stage
    
    rel_tol_fac, updates = self.norm_interface(ug0), []
    if rel_tol_fac < 1e-6: rel_tol_fac = 1.
    for k in range(maxiter):
        # Dirichlet time-integration
        u1 = np.copy(u10)
        ug_WF_func = self.interpolation(t2, ug_WF_old)
        
        u1_list = [] ## storage for initial flux computation
        for i, t in enumerate(t1[:-1]):
            u1, _, flux_WF_1[i+1], flux_WF_2[i+1] = self.solve_dirichlet_SDIRK2(t, dt1, u1, ug_WF_func)
            if i < 2: ## only for up to 2 steps
                u1_list.append(np.copy(u1))
        ## compute initial flux
        if len(u1_list) == 1: # only single timestep, use same approximation as for IE
            u1dot = (-u10 + u1_list[0])/dt1
            ugdot = (-ug0 + ug_WF_func(dt1))/dt1
            flux0 = self.Mgg1.dot(ugdot) + self.Mg1.dot(u1dot) + self.Agg1.dot(ug0) + self.Ag1.dot(u10)             
        else: ## at least 2 timesteps, use 3 point finite difference
            u1dot = (-3*u10 + 4*u1_list[0] - u1_list[1])/(2*dt1)
            ugdot = (-3*ug0 + 4*ug_WF_func(dt1) - ug_WF_func(2*dt1))/(2*dt1)
            flux0 = self.Mgg1.dot(ugdot) + self.Mg1.dot(u1dot) + self.Agg1.dot(ug0) + self.Ag1.dot(u10) 
        
        # Neumann time-integration
        u2 = np.copy(u20)
        flux_WF_1[0], flux_WF_2[0] = flux0, flux0 ## update initial flux
        
        flux_WF_1_func = self.interpolation(t1_stage, flux_WF_1)
        flux_WF_2_func = self.interpolation(t1, flux_WF_2)
        for i, t in enumerate(t2[:-1]):
            u2, ug_WF_new[i+1], _ = self.solve_neumann_SDIRK2(t, dt2, u2, ug_WF_new[i], flux_WF_1_func, flux_WF_2_func)
            
        # relaxation
        tmp = np.copy(ug_WF_old[-1]) # backup of old last value
        # new becomes old in next iteration
        ug_WF_old = [theta*ug_WF_new[i] + (1-theta)*ug_WF_old[i] for i in range(len(t2))]
        
        # bookkeeping
        updates.append(self.norm_interface(ug_WF_old[-1] - tmp))
        if updates[-1]/rel_tol_fac < TOL: # STOPPING CRITERIA FOR FIXED POINT ITERATION
            break
    ## last output only for work performance test, include this for other ones as well or remove afterwards?
    return u1, u2, ug_WF_old[-1], updates, k+1

def solve_dirichlet_SDIRK2(self, t, dt, uold, ug_WF_f): 
    dta = self.a*dt
    ug_old, ug_new, ug_mid = ug_WF_f(t), ug_WF_f(t + dt), ug_WF_f(t + dta)
    ug_dot_a = (ug_mid - ug_old)/dta
    ug_dot_new = (ug_new - (ug_old + (1 - self.a)*dt*ug_dot_a))/dta
    
    # stage 1
    s1 = uold
    U1 = self.linear_solver(self.M1 + dta*self.A1, 
                            self.M1.dot(s1) - dta*(self.M1g.dot(ug_dot_a) + self.A1g.dot(ug_mid)))
    k1 = (U1 - s1)/dta
    
    # stage 2
    s2 = uold + dt*(1 - self.a)*k1
    U2 = self.linear_solver(self.M1 + dta*self.A1, 
                            self.M1.dot(s2) - dta*(self.M1g.dot(ug_dot_new) + self.A1g.dot(ug_new)))
    k2 = (U2 - s2)/dta
    
    localu = dt*((self.a - self.ahat)*k2 + (self.ahat - self.a)*k1)

    flux_1 = self.Mgg1.dot(ug_dot_a) + self.Mg1.dot(k1) + self.Agg1.dot(ug_mid) + self.Ag1.dot(U1)
    flux_2 = self.Mgg1.dot(ug_dot_new) + self.Mg1.dot(k2) + self.Agg1.dot(ug_new) + self.Ag1.dot(U2)    
        
    return U2, localu, flux_1, flux_2

def solve_neumann_SDIRK2(self, t, dt, uold, ug_old, flux1, flux2):
    dta = self.a*dt
    n = len(ug_old)
    
    uold_full = np.hstack((uold, ug_old))
    
    s1 = uold_full # stage 1
    b = self.Neumann_M.dot(s1)
    b[-n:] -= dta*flux1(t + dta)
    U1 = self.linear_solver(self.Neumann_M + dta*self.Neumann_A, b)
    k1 = (U1 - s1)/dta
    
    s2 = s1 + dt*(1 - self.a)*k1 # stage 2
    b = self.Neumann_M.dot(s2)
    b[-n:] -= dta*flux2(t + dt)
    U2 = self.linear_solver(self.Neumann_M + dta*self.Neumann_A, b)
    k2 = (U2 - s2)/dta
    
    localu = dt*((self.a - self.ahat)*k2 + (self.ahat - self.a)*k1)
    return U2[:-n], U2[-n:], localu

if __name__ == '__main__':
    from FSI_verification import get_problem, get_solver, solve_monolithic
    from FSI_verification import get_parameters, get_init_cond
    from FSI_verification import verify_with_monolithic, verify_comb_error, verify_splitting_error, verify_test, verify_MR_comb
    
    p_base = get_parameters('test') ## basic testing parameters
    init_cond_1d = get_init_cond(1)
    init_cond_2d = get_init_cond(2)
    p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': 2, 'WR_type': 'DNWR', **p_base}
    p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': 2, 'WR_type': 'DNWR', **p_base}
    
    save = 'verify/DNWR/SDIRK2/'
    ## 1D
    verify_with_monolithic(k = 14, savefig = save, **p1) # ex sol in single iteration
    verify_with_monolithic(k = 16, theta = 0.7, savefig = save + 'non_opt_theta', **p1) # theta not optimal to see actual convergence
    verify_splitting_error(k = 10, savefig = save, **p1)
    verify_comb_error(k = 10, savefig = save, **p1)
    verify_MR_comb(k = 8, savefig = save, **p1)
    # 2D
    verify_with_monolithic(k = 12, savefig = save, **p2)
    verify_splitting_error(k = 10, savefig = save, **p2)
    verify_comb_error(k = 9, savefig = save, **p2)
    verify_MR_comb(k = 8, savefig = save, **p2)
    
    for which in ['air_water', 'air_steel', 'water_steel']:
        pp = get_parameters(which)
        p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': 2, 'WR_type': 'DNWR', **pp, 'tf': 10000}
        p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': 2, 'WR_type': 'DNWR', **pp, 'tf': 10000}
        verify_MR_comb(k = 6, savefig = save + which + '/' + which, **p1)
        verify_MR_comb(k = 6, savefig = save + which + '/' + which, **p2)