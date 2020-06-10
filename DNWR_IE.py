#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:12:13 2020

@author: Peter Meisrimel, Lund University
"""
import numpy as np

#############################################################
#  DNWR IMPLICIT EULER, Multirate
#############################################################
def solve_dirichlet_IE(self, t, dt, uold, ug_WF_f): 
    ug_old, ug_new = ug_WF_f(t), ug_WF_f(t + dt)
    ug_dot_new = (ug_new - ug_old)/dt
    unew = self.linear_solver(self.M1 + dt*self.A1,
                              self.M1.dot(uold) - dt*(self.M1g.dot(ug_dot_new) + self.A1g.dot(ug_new)))
    flux = self.Mg1.dot((unew - uold)/dt) + self.Ag1.dot(unew) + self.Mgg1.dot(ug_dot_new) + self.Agg1.dot(ug_new)
    return unew, flux

def solve_neumann_IE(self, t, dt, uold, ug_old, flux):
    n = len(ug_old)
        
    b = self.Neumann_M.dot(np.hstack((uold, ug_old)))
    b[-n:] -= dt*flux(t + dt)
    unew = self.linear_solver(self.Neumann_M + dt*self.Neumann_A, b)
    return unew[:-n], unew[-n:]

def DNWR_IE(self, tf, N1, N2, init_cond, theta = None, maxiter = 100, TOL = 1e-8):
    if self.WR_type != 'DNWR': raise ValueError('invalid solution method')
    dt1, dt2 = tf/N1, tf/N2
    ## get theta_opt if no theta is given
    if theta is None: theta = self.DNWR_theta_opt(dt1, dt2)
    
    u10, u20, ug0 = self.get_initial_values(init_cond) ## different for 1D and 2D
    flux0 = np.zeros(ug0.shape) ## placeholder only, will be overwritten
    
    t1, t2 = np.linspace(0, tf, N1+1), np.linspace(0, tf, N2+1) # time grids
    ug_WF_old = [np.copy(ug0) for i in range(N2 + 1)] # waveform of ug, N grid
    ug_WF_new = [np.copy(ug0) for i in range(N2 + 1)] # waveform of ug, N grid
    flux_WF = [np.copy(flux0) for i in range(N1 + 1)] # waveform of fluxes, D grid
   
    rel_tol_fac, updates = self.norm_L2(ug0), []
    if rel_tol_fac < 1e-6: rel_tol_fac = 1. ## safeguard again too small update factor
    for k in range(maxiter):
        # Dirichlet time-integration
        u1 = np.copy(u10)
        ug_WF_func = self.interpolation(t2, ug_WF_old)
        for i, t in enumerate(t1[:-1]):
            u1, flux_WF[i+1] = self.solve_dirichlet_IE(t, dt1, u1, ug_WF_func)
            if i == 0: ## compute initial flux
                u1dot = (-u10 + u1)/dt1
                ugdot = (-ug0 + ug_WF_func(dt1))/dt1
                flux_WF[0] = self.Mgg1.dot(ugdot) + self.Mg1.dot(u1dot) + self.Agg1.dot(ug0) + self.Ag1.dot(u10)
            
        # Neumann time-integration
        u2 = np.copy(u20)
        flux_WF_func = self.interpolation(t1, flux_WF)
        for i, t in enumerate(t2[:-1]):
            u2, ug_WF_new[i+1] = self.solve_neumann_IE(t, dt2, u2, ug_WF_new[i], flux_WF_func)
            
        # relaxation
        tmp = np.copy(ug_WF_old[-1]) # backup of old last value
        # new becomes old in next iteration
        ug_WF_old = [theta*ug_WF_new[i] + (1-theta)*ug_WF_old[i] for i in range(len(t2))]
            
        # bookkeeping
        updates.append(self.norm_L2(ug_WF_old[-1] - tmp))
        if updates[-1]/rel_tol_fac < TOL: # STOPPING CRITERIA FOR FIXED POINT ITERATION
            break
    return u1, u2, ug_WF_old[-1], updates, k + 1

if __name__ == '__main__':
    from FSI_verification import get_problem, get_solver, solve_monolithic
    from FSI_verification import get_parameters, get_init_cond
    from FSI_verification import verify_with_monolithic, verify_comb_error, verify_splitting_error, verify_MR_comb
    
    p_base = get_parameters('test') ## basic testing parameters
    init_cond_1d = get_init_cond(1)
    init_cond_2d = get_init_cond(2)
    p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': 1, 'WR_type': 'DNWR', **p_base}
    p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': 1, 'WR_type': 'DNWR', **p_base}
    
    save = 'verify/DNWR/IE/'
    ## 1D
    verify_with_monolithic(k = 14, savefig = save, **p1) # ex sol in single iteration
    verify_with_monolithic(k = 14, theta = 0.7, savefig = save + 'non_opt_theta', **p1) # theta not optimal to see actual convergence
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
        p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': 1, 'WR_type': 'DNWR', **pp, 'tf': 1000}
        p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': 1, 'WR_type': 'DNWR', **pp, 'tf': 1000}
        verify_MR_comb(k = 10, savefig = save + which + '/' + which, **p1)
        verify_MR_comb(k = 8, savefig = save + which + '/' + which, **p2)
    
    ## EXTRA LEN STUFF    
    p_base = get_parameters('test') ## basic testing parameters
    p_len = {'len_1' : 2, 'len_2': 3}
    init_cond_1d_len = get_init_cond(1, True)
    init_cond_2d_len = get_init_cond(2, True)
    p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': 1, 'WR_type': 'DNWR', **p_base, **p_len}
    p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': 1, 'WR_type': 'DNWR', **p_base, **p_len}
    
    save = 'verify/DNWR/IE/extra_len/'
    ## 1D
    verify_with_monolithic(k = 14, savefig = save, **p1) # ex sol in single iteration
    verify_with_monolithic(k = 14, theta = 0.7, savefig = save + 'non_opt_theta', **p1) # theta not optimal to see actual convergence
    verify_splitting_error(k = 10, savefig = save, **p1)
    verify_comb_error(k = 10, savefig = save, **p1)
    verify_MR_comb(k = 8, savefig = save, **p1)
    ## 2D
    verify_with_monolithic(k = 12, savefig = save, **p2)
    verify_splitting_error(k = 10, savefig = save, **p2)
    verify_comb_error(k = 9, savefig = save, **p2)
    verify_MR_comb(k = 8, savefig = save, **p2)
    
    for which in ['air_water', 'air_steel', 'water_steel']:
        pp = get_parameters(which)
        p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': 1, 'WR_type': 'DNWR', **pp, 'tf': 1000}
        p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': 1, 'WR_type': 'DNWR', **pp, 'tf': 1000}
        verify_MR_comb(k = 8, savefig = save + which + '/' + which, **p1)
        verify_MR_comb(k = 8, savefig = save + which + '/' + which, **p2)