#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:33:03 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np

###############
# double adaptive is deprecated
# initial stepsizes might be off, due to length variables only stored locally for NNWR
###############

#############################################################
#  SDIRK 2, NNWR, time-adaptive
"""
single adaptive, i.e. use potentially different time-grids on the different domains in the Dirichlet solve
use same time-grids in the Neumann stage to calculate phi
"""
#############################################################
def NNWR_SDIRK2_TA_single(self, tf, init_cond, maxiter = 20, TOL = 1e-8):
    if self.WR_type != 'NNWR': raise ValueError('invalid solution method')
    if self.comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
    TOL_D = TOL/5 ## paper
    
    u10, u20, ug0 = self.get_initial_values(init_cond) ## different for 1D and 2D
    if self.ID_SELF == 0:  uD0, uN0 = u10, np.zeros(u20.shape)
    else:                  uD0, uN0 = u20, np.zeros(u10.shape)
    
    tt_old, g_old = [0., tf], [np.copy(ug0), np.copy(ug0)]
    
    ## which = 1 for both, since M1 will always be the mass matrix corresponding to uD
    dt0 = self.get_dt0(tf, TOL_D, uD0, which = 1)
    
    flux0 = np.zeros(ug0.shape) ## placeholder
    timesteps = 0 # total number of timesteps
    
    rel_tol_fac, updates = self.norm_interface(ug0), []
    if rel_tol_fac < 1e-6: rel_tol_fac = 1.
    self.comm.Barrier()
    for k in range(maxiter):
        # Dirichlet time-integration
        t, uD = 0., np.copy(uD0) # init time-integration
        dt = dt0
        tt_stage, tt_new = [0.], [0.]
        flux_self_WF, flux_self_WF_stage = [np.copy(flux0)], [np.copy(flux0)] # flux WF
        
        self.r_old = TOL_D ## for time adaptivity
        g_WF_func = self.interpolation(tt_old, g_old) # interface values WF
        
        u1_list = [] # initial flux computation
        while t + dt < tf:
            uD, err, flux_stage, flux = self.solve_dirichlet_SDIRK2(t, dt, uD, g_WF_func)
            tt_stage.append(t + self.a*dt); flux_self_WF_stage.append(flux_stage)
            t += dt
            tt_new.append(t); flux_self_WF.append(flux)
            if len(u1_list) < 2: ## storage for initial flux computation
                u1_list.append((dt, np.copy(uD)))
            dt = self.get_new_dt_PI(dt, self.norm_inner(err, 'D'), TOL_D)
            if dt < 1e-14: raise ValueError('too small timesteps, aborting')
        # final timestep
        dt = tf - t
        uD, _, flux1, flux2 = self.solve_dirichlet_SDIRK2(t, dt, uD, g_WF_func)
        tt_stage.append(t + self.a*dt); flux_self_WF_stage.append(flux_stage)
        tt_new.append(tf); flux_self_WF.append(flux)
        if len(u1_list) < 2: ## storage for initial flux computation
            u1_list.append((dt, np.copy(uD)))
            
        if len(u1_list) == 1: ## only single timestep was done => 2 point difference
            u1dot = (-uD0 + uD)/dt0
            ugdot = (-ug0 + g_WF_func(dt0))/dt0
            flux0 = self.Mgg1.dot(ugdot) + self.Mg1.dot(u1dot) + self.Agg1.dot(ug0) + self.Ag1.dot(uD0)
        elif len(u1_list) == 2:
            ## check paper for origin of this formula
            dtt1, dtt2 = dt0, u1_list[1][0]
            c = dtt1/(dtt1 + dtt2)
            u1dot = (-uD0*(1 - c**2) + u1_list[0][1] - u1_list[1][1]*c**2)/(dtt1*(1 - c))
            ugdot = (-ug0*(1 - c**2) + g_WF_func(dtt1) - g_WF_func(dtt1 + dtt2)*c**2)/(dtt1*(1 - c))
            flux0 = self.Mgg1.dot(ugdot) + self.Mg1.dot(u1dot) + self.Agg1.dot(ug0) + self.Ag1.dot(uD0)
        else:
            raise ValueError('this is not supposed to happen, check code in detail')
        flux_self_WF[0], flux_self_WF_stage[0] = flux0, flux0
        
        ## Communication
        tt_other = self.comm.sendrecv(tt_new, dest = self.ID_OTHER, source = self.ID_OTHER)
        tt_other_stage = [0.] + [tt_other[i] + self.a*(tt_other[i+1] - tt_other[i]) for i in range(len(tt_other) - 1)]
        flux_other_WF_stage = self.comm.sendrecv(flux_self_WF_stage, dest = self.ID_OTHER, source = self.ID_OTHER)
        flux_other_WF = self.comm.sendrecv(flux_self_WF, dest = self.ID_OTHER, source = self.ID_OTHER)
        
        ## interpolation of fluxes for Neumann stage
        f_flux_other_stage = self.interpolation(tt_other_stage, flux_other_WF_stage)
        f_flux_other = self.interpolation(tt_other, flux_other_WF)
        
        ## summing up fluxes
        for i, t in enumerate(tt_stage):
            flux_self_WF_stage[i] = -flux_self_WF_stage[i] - f_flux_other_stage(t)
        for i, t in enumerate(tt_new):
            flux_self_WF[i] = -flux_self_WF[i] - f_flux_other(t)
        
        # Neumann time-integration
        phiN, phi_self = np.copy(uN0), [np.zeros(ug0.shape)]
        f_flux_WF_stage = self.interpolation(tt_stage, flux_self_WF_stage)
        f_flux_WF = self.interpolation(tt_new, flux_self_WF)
        ## non adaptive for neumann
        for i, t in enumerate(tt_new[:-1]):
            phiN, phiG, _ = self.solve_neumann_SDIRK2(t, tt_new[i+1] - t, phiN, phi_self[-1], f_flux_WF_stage, f_flux_WF)
            phi_self.append(phiG)
        
        phi_other = self.comm.sendrecv(phi_self, dest = self.ID_OTHER, source = self.ID_OTHER)
        phi_other_f = self.interpolation(tt_other, phi_other)
        
        # relaxation, interpolate old g to new grid
        tmp = np.copy(g_old[-1]) # backup of old last value
        ## calculate optimal relaxation parameter
        theta = self.NNWR_theta_opt_TA(tt_new, tt_other)
        g_old = g_WF_func(tt_new) # interpolate old g_old to new grid
        g_old = [g_old[i] - theta*(phi_self[i] + phi_other_f(t)) for i, t in enumerate(tt_new)]
        tt_old = tt_new
        
        # bookkeeping
        updates.append(self.norm_interface(g_old[-1] - tmp))
        print(self.ID_SELF, k, updates[-1], len(tt_new) - 1)
        timesteps += 2*(len(tt_new) + len(tt_other) - 2)
        if updates[-1]/rel_tol_fac < TOL: # STOPPING CRITERIA FOR FIXED POINT ITERATION
            break
    ## only return results on first process
    if self.ID_SELF == 0:
        uD_other = self.comm.recv(source = self.ID_OTHER, tag = self.TAG_DATA)
        return uD, uD_other, g_old[-1], updates, k+1, timesteps
    else:
        self.comm.send(uD, dest = self.ID_OTHER, tag = self.TAG_DATA)
        
#############################################################
#  SDIRK 2, NNWR, time-adaptive
"""
double adaptive, different grids for the both domains and also the correction stage
rather experimental, not 100% sure if this makes too much sense
DEPRECATED!
"""
#############################################################
def NNWR_SDIRK2_TA_double(self, tf, init_cond, maxiter = 100, TOL = 1e-8, flux0_init = False):
    print('method is deprecated')
    if self.WR_type != 'NNWR': raise ValueError('invalid solution method')
    if self.comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
    TOL_D = TOL/5 ## paper
    
    u10, u20, ug0 = self.get_initial_values(init_cond) ## different for 1D and 2D
    if self.ID_SELF == 0:  uD0, uN0 = u10, np.zeros(u20.shape)
    else:                  uD0, uN0 = u20, np.zeros(u10.shape)
    
    tt_old, g_old = [0., tf], [np.copy(ug0), np.copy(ug0)]
    
    dt0_self = self.get_dt0(tf, TOL_D, uD0, which = 1)
    dt0_other = self.comm.sendrecv(dt0_self, dest = self.ID_OTHER, source = self.ID_OTHER)
    
    if (not flux0_init) and abs(dt0_self - dt0_other) > 1e-14:
        dt0 = min(dt0_self, dt0_other)
        uD1, uD2, ug1, ug2, flux0_steps = self.NNWR_SDIRK2(2*dt0, 2, 2, init_cond, None, maxiter, TOL, flux0_init = True)
        uDdot = (-3*uD0 + 4*uD1 - uD2)/(2*dt0)
        ugdot = (-3*ug0 + 4*ug1 - ug2)/(2*dt0)
        flux0 = self.Mgg1.dot(ugdot) + self.Mg1.dot(uDdot) + self.Agg1.dot(ug0) + self.Ag1.dot(uD0) 
    else:
        flux0 = np.zeros(ug0.shape)
        flux0_steps = 0
    timesteps = flux0_steps*2
    
    rel_tol_fac, updates = self.norm_interface(ug0), []
    if rel_tol_fac < 1e-6: rel_tol_fac = 1.
    self.comm.Barrier()
    for k in range(maxiter):
        # Dirichlet time-integration
        t, uD = 0., np.copy(uD0) # init time-integration
        dt = dt0_self
        tt_stage, tt_new = [0.], [0.]
        flux_self_WF, flux_self_WF_stage = [np.copy(flux0)], [np.copy(flux0)] # flux WF
        
        self.r_old = TOL_D ## for time adaptivity
        g_WF_func = self.interpolation(tt_old, g_old) # interface values WF
        while t + dt < tf:
            uD, err, flux_stage, flux = self.solve_dirichlet_SDIRK2(t, dt, uD, g_WF_func)
            tt_stage.append(t + self.a*dt); flux_self_WF_stage.append(flux_stage)
            t += dt
            tt_new.append(t); flux_self_WF.append(flux)
            dt = self.get_new_dt_PI(dt, err, TOL_D)
            if dt < 1e-14: raise ValueError('too small timesteps, aborting')
        # final timestep
        dt = tf - t
        uD, _, flux1, flux2 = self.solve_dirichlet_SDIRK2(t, dt, uD, g_WF_func)
        tt_stage.append(t + self.a*dt); flux_self_WF_stage.append(flux_stage)
        tt_new.append(tf); flux_self_WF.append(flux)
        
        ## Communication
        tt_other = self.comm.sendrecv(tt_new, dest = self.ID_OTHER, source = self.ID_OTHER)
        tt_other_stage = [0.] + [tt_other[i] + self.a*(tt_other[i+1] - tt_other[i]) for i in range(len(tt_other) - 1)]
        flux_other_WF_stage = self.comm.sendrecv(flux_self_WF_stage, dest = self.ID_OTHER, source = self.ID_OTHER)
        flux_other_WF = self.comm.sendrecv(flux_self_WF, dest = self.ID_OTHER, source = self.ID_OTHER)
        
        ## interpolation of fluxes for Neumann stage
        f_flux_other_stage = self.interpolation(tt_other_stage, flux_other_WF_stage)
        f_flux_other = self.interpolation(tt_other, flux_other_WF)
        
        ## summing up fluxes
        for i, t in enumerate(tt_stage):
            flux_self_WF_stage[i] = -flux_self_WF_stage[i] - f_flux_other_stage(t)
        for i, t in enumerate(tt_new):
            flux_self_WF[i] = -flux_self_WF[i] - f_flux_other(t)
        
        # Neumann time-integration
        phiN, phi_self = np.copy(uN0), [np.zeros(ug0.shape)]
        f_flux_WF_stage = self.interpolation(tt_stage, flux_self_WF_stage)
        f_flux_WF = self.interpolation(tt_new, flux_self_WF)
        # adaptive for neumann as well
        t, tt_neu = 0., [0.]
        dt = dt0_self ## same dt0
        while t + dt < tf:
            phiN, phiG, err = self.solve_neumann_SDIRK2(t, dt, phiN, phi_self[-1], f_flux_WF_stage, f_flux_WF)
            t += dt
            tt_neu.append(t); phi_self.append(phiG)
            dt = self.get_new_dt_PI(dt, err, TOL_D)
            if dt < 1e-14: raise ValueError('too small timesteps, aborting')
        # final timestep
        dt = tf - t
        phiN, phiG, _ = self.solve_neumann_SDIRK2(t, dt, phiN, phi_self[-1], f_flux_WF_stage, f_flux_WF)
        tt_neu.append(tf); phi_self.append(phiG)
            
        ## communication
        phi_other = self.comm.sendrecv(phi_self, dest = self.ID_OTHER, source = self.ID_OTHER)
        tt_neu_other = self.comm.sendrecv(tt_neu, dest = self.ID_OTHER, source = self.ID_OTHER)
        phi_other_f = self.interpolation(tt_neu_other, phi_other)
        
        phi_self_f = self.interpolation(tt_neu, phi_self)
        
        # relaxation, interpolate old g to new grid
        tmp = np.copy(g_old[-1]) # backup of old last value
        ## calculate optimal relaxation parameter
        theta = self.NNWR_theta_opt_TA(tt_new, tt_other)
        ## g_WF_func already defined earlier 
        g_old = [g_WF_func(t) - theta*(phi_self_f(t) + phi_other_f(t)) for i, t in enumerate(tt_new)]
        tt_old = tt_new
        
        # bookkeeping
        updates.append(self.norm_interface(g_old[-1] - tmp))
        print(self.ID_SELF, k, updates[-1], len(tt_new) - 1, len(tt_other) - 1)
        timesteps += len(tt_new) + len(tt_other) + len(tt_neu) + len(tt_neu_other) - 4
        if updates[-1]/rel_tol_fac < TOL: # STOPPING CRITERIA FOR FIXED POINT ITERATION
            break
    ## only return results on first process
    if self.ID_SELF == 0:
        uD_other = self.comm.recv(source = self.ID_OTHER, tag = self.TAG_DATA)
        return uD, uD_other, g_old[-1], updates, k+1, timesteps
    else:
        self.comm.send(uD, dest = self.ID_OTHER, tag = self.TAG_DATA)
        
if __name__ == '__main__':
    from FSI_verification import get_problem, get_solver, solve_monolithic
    from FSI_verification import get_parameters, get_init_cond
    from FSI_verification import verify_adaptive, visual_verification
    
    ## mpiexec -n 2 python3 NNWR_SDIRK2_TA.py
    
    p_base = get_parameters('test') ## basic testing parameters
    init_cond_1d = get_init_cond(1)
    init_cond_2d = get_init_cond(2)
    save = 'verify/NNWR/TA/'
    
    for order in [-1]: # single
        p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': order, 'WR_type': 'NNWR', **p_base}
        p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': order, 'WR_type': 'NNWR', **p_base}
    
        verify_adaptive(k = 8, which_ref = 'fine', savefig = save, **p1) # 1D
        verify_adaptive(k = 7, which_ref = 'fine', savefig = save, **p2) # 2D
        
        for which in ['air_water', 'air_steel', 'water_steel']:
            pp = get_parameters(which)
            p1 = {'init_cond': init_cond_1d, 'n': 50, 'dim': 1, 'order': order, 'WR_type': 'NNWR', **pp, 'tf': 1000}
            p2 = {'init_cond': init_cond_2d, 'n': 32, 'dim': 2, 'order': order, 'WR_type': 'NNWR', **pp, 'tf': 1000}
            verify_adaptive(k = 7, savefig = save + which + '/' + which, **p1, which_ref = 'fine')
            verify_adaptive(k = 6, savefig = save + which + '/' + which, **p2, which_ref = 'fine')