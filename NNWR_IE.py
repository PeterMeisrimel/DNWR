#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:31:12 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np

#############################################################
#  IMPLICIT EULER, NNWR
#############################################################
def NNWR_IE(self, tf, N1, N2, init_cond, theta = None, maxiter = 100, TOL = 1e-8):
    if self.WR_type != 'NNWR': raise ValueError('invalid solution method')
    if self.comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
    
    u10, u20, ug0 = self.get_initial_values(init_cond) ## different for 1D and 2D
    ## important, initial values are ordered, each solver needs to determine which is which
    if self.ID_SELF == 0:
        uD0, uN0 = u10, np.zeros(u20.shape)
        N_self, N_other = N1, N2
    else:
        uD0, uN0 = u20, np.zeros(u10.shape)
        N_self, N_other = N2, N1
    dt, dt_other = tf/N_self, tf/N_other
    
    t_self, t_other = np.linspace(0, tf, N_self + 1), np.linspace(0, tf, N_other + 1)
    g_old = [np.copy(ug0) for i in range(N_self + 1)]
    phi_self = [np.zeros(ug0.shape) for i in range(N_self + 1)]
        
    flux0 = np.zeros(ug0.shape) ## placeholder
    flux_WF = [np.copy(flux0) for i in range(N_self + 1)]
    
    if theta is None: theta = self.NNWR_theta_opt(dt, dt_other)
    
    rel_tol_fac, updates = self.norm_L2(ug0), []
    if rel_tol_fac < 1e-6: rel_tol_fac = 1.
    self.comm.Barrier()
    for k in range(maxiter):
        ## Dirichlet
        uD = np.copy(uD0)
        g_WF_func = self.interpolation(t_self, g_old)
        for i, t in enumerate(t_self[:-1]):
            uD, flux_WF[i+1] = self.solve_dirichlet_IE(t, dt, uD, g_WF_func)
            if i == 0: ## compute initial flux
                u1dot = (-uD0 + uD)/dt
                ugdot = (-ug0 + g_WF_func(dt))/dt
                flux_WF[0] = self.Mgg1.dot(ugdot) + self.Mg1.dot(u1dot) + self.Agg1.dot(ug0) + self.Ag1.dot(u10)
            
        ## communication
        flux_WF_other = self.comm.sendrecv(flux_WF, dest = self.ID_OTHER, source = self.ID_OTHER)
        # interpolation of fluxes to own grid
        f_flux_WF_other = self.interpolation(t_other, flux_WF_other)
        for i, t in enumerate(t_self):
            flux_WF[i] = -flux_WF[i] - f_flux_WF_other(t)
            
        ## Neumann
        uN = np.copy(uN0)
        flux_WF_func = self.interpolation(t_self, flux_WF)
        for i, t in enumerate(t_self[:-1]):
            uN, phi_self[i+1] = self.solve_neumann_IE(t, dt, uN, phi_self[i], flux_WF_func)
        
        ## communication
        phi_other = self.comm.sendrecv(phi_self, dest = self.ID_OTHER, source = self.ID_OTHER)
        phi_other_f = self.interpolation(t_other, phi_other)
        
        # relaxation, interpolate old g to new grid
        tmp = np.copy(g_old[-1]) # backup of old last value
        for i, t in enumerate(t_self):
            g_old[i] = g_old[i] - theta*(phi_self[i] + phi_other_f(t))
            
        # bookkeeping
        updates.append(self.norm_L2(g_old[-1] - tmp))
        print(self.ID_SELF, k, updates[-1])
        if updates[-1]/rel_tol_fac < TOL: # STOPPING CRITERIA FOR FIXED POINT ITERATION
            break
    ## only return results on first process
    if self.ID_SELF == 0:
        uD_other = self.comm.recv(source = self.ID_OTHER, tag = self.TAG_DATA)
        return uD, uD_other, g_old[-1], updates, k+1
    else:
        self.comm.send(uD, dest = self.ID_OTHER, tag = self.TAG_DATA)
        
if __name__ == '__main__':
    ## mpiexec -n 2 python3 NNWR_IE.py 
    from FSI_verification import get_problem, get_solver, solve_monolithic
    from FSI_verification import get_parameters, get_init_cond
    from FSI_verification import verify_with_monolithic, verify_comb_error, verify_splitting_error, verify_MR_comb
    
    p_base = get_parameters('test') ## basic testing parameters
    init_cond_1d = get_init_cond(1)
    init_cond_2d = get_init_cond(2)
    p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': 1, 'WR_type': 'NNWR', **p_base}
    p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': 1, 'WR_type': 'NNWR', **p_base}
    save = 'verify/NNWR/IE/'
        
    ## 1D
    verify_with_monolithic(k = 14, savefig = save, **p1) # ex sol in single iteration
    verify_with_monolithic(k = 14, theta = 0.3, savefig = save + 'non_opt_theta', **p1) # theta not optimal to see actual convergence
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
        p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': 1, 'WR_type': 'NNWR', **pp, 'tf': 1000}
        p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': 1, 'WR_type': 'NNWR', **pp, 'tf': 1000}
        verify_MR_comb(k = 8, savefig = save + which + '/' + which, **p1)
        verify_MR_comb(k = 8, savefig = save + which + '/' + which, **p2)