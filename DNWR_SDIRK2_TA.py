#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:28:39 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np

#############################################################
#  SDIRK 2, Time adaptive, single adaptive
#############################################################    
## time grid is constructed during Dirichlet solve and same time grid is used for Neumann solver.
## Implementation not properly maintained, use on own risk
def DNWR_SDIRK2_TA_single(self, tf, init_cond, maxiter = 100, TOL = 1e-8):
    print('this scheme is outdated and did not receive analogous updates')
    print(f'TOL = {TOL}')
    if self.WR_type != 'DNWR': raise ValueError('invalid solution method')
    u10, u20, ug0 = self.get_initial_values(init_cond) ## different for 1D and 2D
    
    tt_old = [0., tf]
    ug_WF_old = [np.copy(ug0), np.copy(ug0)] # waveform of ug, N grid
    ug_WF_new = [np.copy(ug0), np.copy(ug0)] # waveform of ug, N grid
    flux0 = np.zeros(ug0.shape) # zero flux irrelevant
        
    timesteps = 0
    TOL_FP, TOL_D = TOL, TOL/5 ## paper
    dt0 = self.get_dt0(tf, TOL_D, u10, which = 1)
    rel_tol_fac, updates = self.norm_L2(ug0), []
    if rel_tol_fac < 1e-6: rel_tol_fac = 1.
    self.r_old = TOL_D # for PI controller
    for k in range(maxiter):
        # Dirichlet time-integration
        t, u1 = 0., np.copy(u10) # init time-integration
        dt = dt0
        tt, tt_stage = [0.], [0.] # times for flux WF
        flux_WF_1, flux_WF_2 = [np.copy(flux0)], [np.copy(flux0)] # data for flux WF
        
        ug_WF_func = self.interpolation(tt_old, ug_WF_old) # interface values WF
        while t + dt < tf:
            u1, err1, flux1, flux2 = self.solve_dirichlet_SDIRK2(t, dt, u1, ug_WF_func)
            tt_stage.append(t + self.a*dt); flux_WF_1.append(flux1)
            t += dt
            tt.append(t); flux_WF_2.append(flux2)
            dt = self.get_new_dt_PI(dt, err1, TOL_D)
            if dt < 1e-14: raise ValueError('too small timesteps, aborting')
        # final timestep
        dt = tf - t
        u1, _, flux1, flux2 = self.solve_dirichlet_SDIRK2(t, dt, u1, ug_WF_func)
        tt_stage.append(t + self.a*dt); flux_WF_1.append(flux1)
        tt.append(tf); flux_WF_2.append(flux2)
        
        # Neumann time-integration
        u2 = np.copy(u20)
        flux_WF_1_func = self.interpolation(tt_stage, flux_WF_1)
        flux_WF_2_func = self.interpolation(tt, flux_WF_2)
        ug_WF_new = [np.copy(ug0) for i in range(len(tt))]
        for i, t in enumerate(tt[:-1]):
            u2, ug_WF_new[i+1], _ = self.solve_neumann_SDIRK2(t, tt[i+1] - t, u2, ug_WF_new[i], flux_WF_1_func, flux_WF_2_func)
        
        # relaxation
        tmp = np.copy(ug_WF_old[-1]) # backup of old last value
        ug_WF_old = ug_WF_func(tt)# interpolate old data to new grid
        ## relaxation, calculate optimal theta
        theta = self.DNWR_theta_opt_TA(tt, tt)
        # new becomes old in next iteration
        ug_WF_old = [theta*ug_WF_new[i] + (1-theta)*ug_WF_old[i] for i in range(len(tt))]
        tt_old = tt
        
        # bookkeeping
        updates.append(self.norm_L2(ug_WF_old[-1] - tmp))
        timesteps += 2*len(tt) - 2
        print(k, updates[-1], len(tt) - 1)
        if updates[-1]/rel_tol_fac < TOL_FP: # STOPPING CRITERIA FOR FIXED POINT ITERATION
            break
    return u1, u2, ug_WF_old[-1], updates, k+1, timesteps

#############################################################
#  SDIRK 2, Time adaptive, double adaptive
#############################################################    
## Does adaptivity on both the Dirichlet and Neumann solve
def DNWR_SDIRK2_TA_double(self, tf, init_cond, maxiter = 100, TOL = 1e-8):
    if self.WR_type != 'DNWR': raise ValueError('invalid solution method')
    print(f'TOL = {TOL}')
    TOL_FP, TOL_D, TOL_N = TOL, TOL/5, TOL/5 ## paper
    
    u10, u20, ug0 = self.get_initial_values(init_cond) ## different for 1D and 2D
    flux0 = np.zeros(ug0.shape) # placeholder only
    
    t1, t2_old = [0., tf], [0., tf] ## initial time grids
    ug_WF_old = [np.copy(ug0), np.copy(ug0)] # waveform of ug, N grid
    ug_WF_new = [np.copy(ug0), np.copy(ug0)] # waveform of ug, N grid
    
    ## intitial timesteps do not change
    dt0_D = self.get_dt0(tf, TOL_D, u10, which = 1)
    dt0_N = self.get_dt0(tf, TOL_N, u20, which = 2)
        
    timesteps = 0 # counter for total number of timesteps
    
    rel_tol_fac, updates = self.norm_L2(ug0), []
    if rel_tol_fac < 1e-6: rel_tol_fac = 1.
    for k in range(maxiter):
        # Dirichlet time-integration
        t, u1 = 0., np.copy(u10) # init time-integration
        dt = dt0_D ## initial timesteps
        t1, t1_stage = [0.], [0.] # time grids
        flux_WF_1, flux_WF_2 = [np.copy(flux0)], [np.copy(flux0)] # data for flux WF, placeholders
        
        self.r_old = TOL_D # for PI controller
        ug_WF_func = self.interpolation(t2_old, ug_WF_old) # interface values WF
        
        j, u1_list = 0, [] # storage for initial flux computation
        ## adaptive time-integration loop, iterate while current timestep does not overstep tf
        while t + dt < tf:
            u1, err1, flux1, flux2 = self.solve_dirichlet_SDIRK2(t, dt, u1, ug_WF_func)
            t1_stage.append(t + self.a*dt); flux_WF_1.append(flux1) ## stage value
            t += dt
            t1.append(t); flux_WF_2.append(flux2) ## value on full timestep
            if j < 2: ## storage for initial flux computation, saves up to 2 values, and their timesteps
                u1_list.append((dt, np.copy(u1)))
                j += 1
            dt = self.get_new_dt_PI(dt, err1, TOL_D) ## PI controller
            if dt < 1e-14: raise ValueError('too small timesteps, aborting') ## prevents too small timesteps
        # final timestep, truncate to hit tf
        dt = tf - t
        u1, _, flux1, flux2 = self.solve_dirichlet_SDIRK2(t, dt, u1, ug_WF_func)
        t1_stage.append(t + self.a*dt); flux_WF_1.append(flux1)
        t1.append(tf); flux_WF_2.append(flux2)
        if j < 2: ## storage for initial flux computation
            u1_list.append((dt, np.copy(u1)))
            j += 1
            
        if j == 1: ## only single timestep was done => 2 point difference
            u1dot = (-u10 + u1)/dt0_D
            ugdot = (-ug0 + ug_WF_func(dt0_D))/dt0_D
            flux0 = self.Mgg1.dot(ugdot) + self.Mg1.dot(u1dot) + self.Agg1.dot(ug0) + self.Ag1.dot(u10)
        elif j == 2: ## 3 point difference formula
            ## variable stepsize 3 point difference formular
            dtt1, dtt2 = u1_list[0][0], u1_list[1][0]
            c = dtt1/(dtt1 + dtt2)
            u1dot = (-u10*(1 - c**2) + u1_list[0][1] - u1_list[1][1]*c**2)/(dtt1*(1 - c))
            ugdot = (-ug0*(1 - c**2) + ug_WF_func(dtt1) - ug_WF_func(dtt1 + dtt2)*c**2)/(dtt1*(1 - c))
            flux0 = self.Mgg1.dot(ugdot) + self.Mg1.dot(u1dot) + self.Agg1.dot(ug0) + self.Ag1.dot(u10)
        else:
            raise ValueError('this is not supposed to happen, check code in detail')
        
        # Neumann time-integration
        t, u2, ug_old = 0., np.copy(u20), np.copy(ug0)
        dt = dt0_N
        
        flux_WF_1[0], flux_WF_2[0] = flux0, flux0 ## inserting initial flux here
        flux_WF_1_func = self.interpolation(t1_stage, flux_WF_1)
        flux_WF_2_func = self.interpolation(t1, flux_WF_2)
        
        self.r_old = TOL_D # for PI controller
        t2_new, ug_WF_new = [0.], [np.copy(ug0)]
        while t + dt < tf:
            u2, ug_new, err2 = self.solve_neumann_SDIRK2(t, dt, u2, ug_old, flux_WF_1_func, flux_WF_2_func)
            t += dt
            t2_new.append(t)
            ug_WF_new.append(np.copy(ug_new))
            ug_old = ug_new
            dt = self.get_new_dt_PI(dt, err2, TOL_N)
            if dt < 1e-14: raise ValueError('too small timesteps, aborting')
        # final timestep
        dt = tf - t
        u2, ug_new, _ = self.solve_neumann_SDIRK2(t, dt, u2, ug_old, flux_WF_1_func, flux_WF_2_func)
        t2_new.append(tf)
        ug_WF_new.append(np.copy(ug_new))
        ## time-integration done
        
        # relaxation
        tmp = np.copy(ug_WF_old[-1]) # backup of old last value
        ug_WF_old = ug_WF_func(t2_new) # interpolate old data to new grid
        ## relaxation, calculate optimal theta
        theta = self.DNWR_theta_opt_TA(t1, t2_new)
        # new becomes old in next iteration
        ug_WF_old = [theta*ug_WF_new[i] + (1-theta)*ug_WF_old[i] for i in range(len(t2_new))]
        t2_old = t2_new
        
        # bookkeeping
        updates.append(self.norm_L2(ug_WF_old[-1] - tmp))
        timesteps += len(t2_new) + len(t1) - 2
        if updates[-1]/rel_tol_fac < TOL_FP: # STOPPING CRITERIA FOR FIXED POINT ITERATION
            print('converged', len(t1) - 1, len(t2_new) - 1)
            break
    return u1, u2, ug_WF_old[-1], updates, k+1, timesteps

if __name__ == '__main__':
    from FSI_verification import get_problem, get_solver, solve_monolithic
    from FSI_verification import get_parameters, get_init_cond
    from FSI_verification import visual_verification, verify_adaptive

    init_cond_1d = get_init_cond(1)
    init_cond_2d = get_init_cond(2)    
    p_base = get_parameters('test') ## basic testing parameters
    save = 'verify/DNWR/TA/'
    
    for order in [-2]: # double adaptive
        p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': order, 'WR_type': 'DNWR', **p_base}
        p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': order, 'WR_type': 'DNWR', **p_base}
    
        verify_adaptive(k = 8, which_ref = 'fine', savefig = save, **p1) # 1D
        verify_adaptive(k = 7, which_ref = 'fine', savefig = save, **p2) # 2D
        
        for which in ['air_water', 'air_steel', 'water_steel']:
            pp = get_parameters(which)
            p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': order, 'WR_type': 'DNWR', **pp, 'tf': 1000}
            p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': order, 'WR_type': 'DNWR', **pp, 'tf': 1000}
            verify_adaptive(k = 9, savefig = save + which + '/' + which, which_ref = 'fine', **p1)
            verify_adaptive(k = 8, savefig = save + which + '/' + which, which_ref = 'fine', **p2)