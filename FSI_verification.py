# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:10:47 2019

@author: Peter Meisrimel, Lund University
"""

import numpy as np
import pylab as pl
from Problem_FSI_1D import Problem_FSI_1D
from Problem_FSI_2D import Problem_FSI_2D
from numpy import sin, pi

pl.rcParams['lines.linewidth'] = 2
pl.rcParams['font.size'] = 14

def get_problem(dim = 1, **kwargs):
    if dim == 1:   return Problem_FSI_1D(**kwargs)
    elif dim == 2: return Problem_FSI_2D(**kwargs)
    else: raise ValueError('invalid dimension')
    
def get_solver(prob, order, WR_type = 'DNWR'):
    ## method for solver selection
    if WR_type == 'DNWR':
        if   order ==  1: return prob.DNWR_IE
        elif order ==  2: return prob.DNWR_SDIRK2
        
        elif order == 22: return prob.DNWR_SDIRK2_test
        
        elif order == -1: return prob.DNWR_SDIRK2_TA_single
        elif order == -2: return prob.DNWR_SDIRK2_TA_double
    elif WR_type == 'NNWR':
        if   order ==  1: return prob.NNWR_IE
        elif order ==  2: return prob.NNWR_SDIRK2
        
        elif order == -1: return prob.NNWR_SDIRK2_TA_single
        elif order == -2: return prob.NNWR_SDIRK2_TA_single
    elif WR_type == 'MONOLITHIC':
        order = abs(order)%10 # to get proper monolithic solver for order = 22, 222
        if   order == 1: return prob.Monolithic_IE
        elif order == 2: return prob.Monolithic_SDIRK2
    else:  raise ValueError('order/method not available')
    
def get_parameters(which):
    ## parameter selection
    lambda_air = 0.0243
    lambda_water = 0.58
    lambda_steel = 48.9
    
    alpha_air = 1.293*1005
    alpha_water = 999.7*4192.1
    alpha_steel = 7836*443
    if which == 'air_water':
        return {'alpha_1': alpha_air, 'alpha_2': alpha_water, 'lambda_1': lambda_air, 'lambda_2': lambda_water}
    elif which == 'air_steel':
        return {'alpha_1': alpha_air, 'alpha_2': alpha_steel, 'lambda_1': lambda_air, 'lambda_2': lambda_steel}
    elif which == 'water_steel':
        return {'alpha_1': alpha_water, 'alpha_2': alpha_steel, 'lambda_1': lambda_water, 'lambda_2': lambda_steel}
    elif which == 'test':
        return {'alpha_1': 1., 'alpha_2': 1., 'lambda_1': 0.1, 'lambda_2': 0.1}
    
def get_init_cond(dim, extra_len = False, num = 1):
    if extra_len:
        ## implicit assertion of len_1 = 2 && len_2 = 3
        if dim == 1:   return lambda x: 500*sin(pi/5*(x+2))
        elif dim == 2: return lambda x, y: 500*sin(pi*y)*sin((pi/5)*(x + 2))
    else:
        if num == 1:
            if dim == 1:   return lambda x: 500*sin(pi/2*(x+1))
            elif dim == 2: return lambda x, y: 500*sin(pi*y)*sin((pi/2)*(x + 1))
        elif num == 2:
            if dim == 2: return lambda x, y: 800*sin(pi*y)*sin(pi*(x + 1))**2
    raise ValueError('no initial cond available for this input')
    
def solve_monolithic(tf = 1, N_steps = 20, init_cond = None, order = 1, **kwargs):
    ## monolithic solver
    prob = get_problem(WR_type = 'MONOLITHIC', **kwargs)
    
    solver = get_solver(prob, order, WR_type = 'MONOLITHIC')
    return solver(tf, N_steps, init_cond)

def ex_sol_grid(n = 30, tf = 1, len_1 = 1, len_2 = 1, ex_sol = None, dim = 2, **kwargs):
    ## evalutes a given ex_sol function on a grid matching the discretization given via "n"
    if dim == 1:
        ref_sol = np.zeros(((len_1 + len_2)*(n + 1) - 1))
        xx = np.linspace(-len_1, len_2, (len_1 + len_2)*(n + 1) + 1)
        for i, x in enumerate(xx[1:-1]):
            ref_sol[i] = ex_sol(x, tf)
        ref1 = ref_sol[:(len_1*(n + 1) - 1)]
        ref2 = ref_sol[-(len_2*(n + 1) - 1):]
        refg = ref_sol[(len_1*(n + 1) - 1):-(len_2*(n + 1) - 1)]
    elif dim == 2:
        ref_sol = np.zeros(n*((len_1 + len_2)*(n + 1) - 1))
        xx = np.linspace(-len_1, len_2, (len_1 + len_2)*(n + 1) + 1)
        yy = np.linspace(0, 1, n + 2)
        for i, x in enumerate(xx[1:-1]):
            for j, y in enumerate(yy[1:-1]):
                ref_sol[i*n + j] = ex_sol(x, y, tf)
        ref1 = ref_sol[:n*(len_1*(n + 1) - 1)]
        ref2 = ref_sol[-n*(len_2*(n + 1) - 1):]
        refg = ref_sol[n*(len_1*(n + 1) - 1):-n*(len_2*(n + 1) - 1)]
    return ref1, ref2, refg

## verifiy order of space discretization by picking a sufficiently fine grid in time and an exact solution
def verify_space_error(tf = 1, init_cond = None, n_min = 2, N_steps = 100, k = 8, ex_sol = None, savefig = None, **kwargs):
    if kwargs['lambda_1'] != kwargs['lambda_2']:    raise ValueError('only works for identical lambda')
    if kwargs['alpha_1']  != kwargs['alpha_2']:     raise ValueError('only works for identical alpha')
        
    errs = {'u1': [], 'u2': [], 'ug': []}
    
    n_list = np.array([n_min*(2**i) for i in range(k)])
    for n in n_list:
        # L2 factor for 2D case, inner points
        L2_fac, L2_fac_intf = 1/(n + 1), np.sqrt(1/(n + 1))
        ref1, ref2, refg = ex_sol_grid(n = n, tf = tf, ex_sol = ex_sol, **kwargs)
    
        u1, u2, ug, flux = solve_monolithic(tf, N_steps, init_cond, order = 2, n = n, **kwargs)
        errs['u1'].append(np.linalg.norm(ref1 - u1, 2)*L2_fac)
        errs['u2'].append(np.linalg.norm(ref2 - u2, 2)*L2_fac)
        errs['ug'].append(np.linalg.norm(refg - ug, 2)*L2_fac_intf)
        n *= 2
    
    pl.figure()
    for k in errs.keys():
        pl.loglog(n_list, errs[k], label = k, marker = 'o')
    pl.loglog(n_list, 1/n_list, label = '1st', linestyle = '--')
    pl.loglog(n_list, 1/(n_list**2), label = '2nd', linestyle = '--')
    pl.legend()
    pl.xlabel('gridsize'); pl.ylabel('err')
    pl.grid(True, which = 'major')
    pl.title('error in space')
    if savefig is not None:
        dim = kwargs['dim']
        s = f'err_space_tf_{tf}_steps_{N_steps}_dim_{dim}.png'
        pl.savefig(savefig + s, dpi = 100)
    print(errs)
    
## verify time-integration order of monolithic solution with itself
def verify_mono_time(tf = 1, init_cond = None, order = 1, k = 5, savefig = None, **kwargs):
    u1_ref, u2_ref, ug_ref, _ = solve_monolithic(tf, 2**(k+1), init_cond, order, **kwargs)
    
    n, dim = kwargs['n'], kwargs['dim']
    if dim == 1:  L2_fac, L2_fac_intf = np.sqrt(1/(n + 1)), 1
    else:         L2_fac, L2_fac_intf = 1/(n + 1), np.sqrt(1/(n + 1))
    
    errs = {'u1': [], 'u2': [], 'ug': []}
    steps = [2**i for i in range(k)]
    for s in steps:
        u1, u2, ug, _ = solve_monolithic(tf, s, init_cond, order, **kwargs)
        errs['u1'].append(np.linalg.norm(u1_ref - u1)*L2_fac)
        errs['u2'].append(np.linalg.norm(u2_ref - u2)*L2_fac)
        errs['ug'].append(np.linalg.norm(ug_ref - ug)*L2_fac_intf)
    for i in range(k-1):
        print(np.log2(errs['u1'][i]/errs['u1'][i+1]),
              np.log2(errs['u2'][i]/errs['u2'][i+1]),
              np.log2(errs['ug'][i]/errs['ug'][i+1]))
        
    pl.figure()
    dts = [tf/s for s in steps]
    pl.loglog(dts, errs['u1'], label = 'u1', marker = 'o')
    pl.loglog(dts, errs['u2'], label = 'u2', marker = 'o')
    pl.loglog(dts, errs['ug'], label = 'ug', marker = 'o')
    pl.loglog(dts, dts, label = '1 st order', linestyle = '--')
    pl.loglog(dts, [t**2 for t in dts], label = '2 nd order', linestyle = '--')
    pl.legend()
    pl.xlabel('dt'); pl.ylabel('Err')
    pl.grid(True, which = 'major')
    pl.title('time-integration error, monolithic')
    if savefig is not None:
        s = f'mono_time_tf_{tf}_ord_{order}_dim_{dim}.png'
        pl.savefig(savefig + s, dpi = 100)

## verify convergence of coupling scheme for decreasing tolerances with monolithic solution, for fixed delta t
def verify_with_monolithic(tf = 1, N_steps = 20, init_cond = None, order = 1, k = 10, theta = None, WR_type = 'DNWR', savefig = None, **kwargs):
    if WR_type == 'NNWR':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
        ID_SELF = comm.rank
    
    prob = get_problem(WR_type = WR_type, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    if WR_type == 'NNWR' and ID_SELF == 1:
        pass
    else:
        u1_ref, u2_ref, ug_ref, _ = solve_monolithic(tf, N_steps, init_cond, order, **kwargs)
    if WR_type == 'NNWR': comm.Barrier() ## sync, only one process calculates monolithic solution
        
    if WR_type == 'NNWR' and ID_SELF == 1:
        tols = [10**(-i) for i in range(k)]
        for tol in tols:
            solver(tf, N_steps, N_steps, init_cond, theta, TOL = tol)
        return None
    ## else: serial DNWR case and ID_SELF == 0 for NNWR
    
    n, dim = kwargs['n'], kwargs['dim']
    if dim == 1:  L2_fac, L2_fac_intf = np.sqrt(1/(n + 1)), 1
    else:         L2_fac, L2_fac_intf = 1/(n + 1), np.sqrt(1/(n + 1))
    
    errs = {'u1': [], 'u2': [], 'ug': [], 'it': []}
    tols = [10**(-i) for i in range(k)]
    for tol in tols:
        u1, u2, ug, E, it = solver(tf, N_steps, N_steps, init_cond, theta, TOL = tol)
        errs['u1'].append(np.linalg.norm(u1_ref - u1)*L2_fac)
        errs['u2'].append(np.linalg.norm(u2_ref - u2)*L2_fac)
        errs['ug'].append(np.linalg.norm(ug_ref - ug)*L2_fac_intf)
        errs['it'].append(it)
    for i in range(k-1):
        print(np.log10(errs['u1'][i]/errs['u1'][i+1]),
              np.log10(errs['u2'][i]/errs['u2'][i+1]),
              np.log10(errs['ug'][i]/errs['ug'][i+1]))
        
    pl.figure()
    pl.loglog(tols, errs['u1'], label = 'u1', marker = 'o')
    pl.loglog(tols, errs['u2'], label = 'u2', marker = 'o')
    pl.loglog(tols, errs['ug'], label = 'ug', marker = 'o')
    pl.loglog(tols, tols, label = '1 st order', linestyle = '--')
    pl.loglog(tols, [t**2 for t in tols], label = '2 nd order', linestyle = '--')
    pl.title('Verification with monolithic solution')
    pl.legend()
    pl.xlabel('TOL'); pl.ylabel('Err')
    pl.grid(True, which = 'major')
    if savefig is not None:
        s = f'verify_mono_time_steps_{N_steps}_dim_{dim}_ord_{order}.png'
        pl.savefig(savefig + s, dpi = 100)
    
    pl.figure()
    pl.semilogx(tols, errs['it'], label = 'iterations', marker = 'o')
    pl.legend()
    pl.xlabel('TOL'); pl.ylabel('DN iter')
    pl.title('Verification with monolithic solution')
    pl.grid(True, which = 'major')
    if savefig is not None:
        s = f'verify_mono_time_iter_steps_{N_steps}_dim_{dim}_ord_{order}.png'
        pl.savefig(savefig + s, dpi = 100)
    print(errs['ug'])
    
## verify "order" of splitting error by calculating it for fixed, small tolerance and increasing number of timesteps
def verify_splitting_error(init_cond, tf = 1, k = 10, kmin = 0, TOL = 1e-12, theta = None, WR_type = 'DNWR', order = 1, savefig = None, **kwargs):
    if WR_type == 'NNWR':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
        ID_SELF = comm.rank
        
    prob = get_problem(WR_type = WR_type, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    if WR_type == 'NNWR' and ID_SELF == 1:
        for n_steps in [2**i for i in range(kmin, k)]:
            comm.Barrier()
            solver(tf, n_steps, n_steps, init_cond, theta, TOL = 1e-13)
        return None
    
    n, dim = kwargs['n'], kwargs['dim']
    if dim == 1:  L2_fac, L2_fac_intf = np.sqrt(1/(n + 1)), 1
    else:         L2_fac, L2_fac_intf = 1/(n + 1), np.sqrt(1/(n + 1))
    
    dts = []
    errs = {'u1': [], 'u2': [], 'ug': [], 'it': []}
    for n_steps in [2**i for i in range(kmin, k)]:
        u1_ref, u2_ref, ug_ref, _ = solve_monolithic(tf, n_steps, init_cond, order, **kwargs)
        if WR_type == 'NNWR':
            comm.Barrier()
        dts.append(tf/n_steps)
        u1, u2, ug, E, it = solver(tf, n_steps, n_steps, init_cond, theta, TOL = 1e-13)
        errs['u1'].append(np.linalg.norm(u1_ref - u1)*L2_fac)
        errs['u2'].append(np.linalg.norm(u2_ref - u2)*L2_fac)
        errs['ug'].append(np.linalg.norm(ug_ref - ug)*L2_fac_intf)
        errs['it'].append(it)
    for i in range(k-1-kmin):
        print(np.log2(errs['u1'][i]/errs['u1'][i+1]),
              np.log2(errs['u2'][i]/errs['u2'][i+1]),
              np.log2(errs['ug'][i]/errs['ug'][i+1]))
        
    pl.figure()
    pl.loglog(dts, errs['u1'], label = 'u1', marker = 'o')
    pl.loglog(dts, errs['u2'], label = 'u2', marker = 'o')
    pl.loglog(dts, errs['ug'], label = 'ug', marker = 'o')
    pl.loglog(dts, dts, label = '1 st order', linestyle = '--')
    pl.loglog(dts, [t**2 for t in dts], label = '2 nd order', linestyle = '--')
    pl.title('Splitting Error')
    pl.legend()
    pl.xlabel('dt'); pl.ylabel('Err')
    pl.grid(True, which = 'major')
    if savefig is not None:
        s = f'splitting_error_dim_{dim}_ord_{order}.png'
        pl.savefig(savefig  + s, dpi = 100)
    
    pl.figure()
    pl.semilogx(dts, errs['it'], label = 'iterations', marker = 'o')
    pl.legend()
    pl.title('Splitting Error')
    pl.xlabel('dt'); pl.ylabel('DN iter')
    pl.grid(True, which = 'major')
    if savefig is not None:
        s = f'splitting_error_iters_dim_{dim}_ord_{order}.png'
        pl.savefig(savefig  + s, dpi = 100)
    
## sum of splitting and time-integration order
## calculating time-integration error( + splitting error) for a small tolerance
def verify_comb_error(init_cond, tf = 1, k = 10, kmin = 0, theta = None, WR_type = 'DNWR', order = 1, savefig = None, **kwargs):
    if WR_type == 'NNWR':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
        ID_SELF = comm.rank
        
    prob = get_problem(WR_type = WR_type, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    if WR_type == 'NNWR' and ID_SELF == 1:
        pass
    else:
        u1_ref, u2_ref, ug_ref, _ = solve_monolithic(tf, 2**(k+1), init_cond, order, **kwargs)
    if WR_type == 'NNWR': comm.Barrier() ## sync, only one process claculates monolithic sol
    
    if WR_type == 'NNWR' and ID_SELF == 1:
        for n_steps in [2**i for i in range(kmin, k)]:
            solver(tf, n_steps, n_steps, init_cond, theta, TOL = 1e-13)
        return None
    
    n, dim = kwargs['n'], kwargs['dim']
    if dim == 1:  L2_fac, L2_fac_intf = np.sqrt(1/(n + 1)), 1
    else:         L2_fac, L2_fac_intf = 1/(n + 1), np.sqrt(1/(n + 1))
    
    dts = []
    errs = {'u1': [], 'u2': [], 'ug': [], 'it': []}
    for n_steps in [2**i for i in range(kmin, k)]:
        dts.append(tf/n_steps)
        u1, u2, ug, E, it = solver(tf, n_steps, n_steps, init_cond, theta, TOL = 1e-13)
        errs['u1'].append(np.linalg.norm(u1_ref - u1)*L2_fac)
        errs['u2'].append(np.linalg.norm(u2_ref - u2)*L2_fac)
        errs['ug'].append(np.linalg.norm(ug_ref - ug)*L2_fac_intf)
        errs['it'].append(it)
    for i in range(k-1-kmin):
        print(np.log2(errs['u1'][i]/errs['u1'][i+1]),
              np.log2(errs['u2'][i]/errs['u2'][i+1]),
              np.log2(errs['ug'][i]/errs['ug'][i+1]))
        
    pl.figure()
    pl.loglog(dts, errs['u1'], label = 'u1', marker = 'o')
    pl.loglog(dts, errs['u2'], label = 'u2', marker = 'o')
    pl.loglog(dts, errs['ug'], label = 'ug', marker = 'o')
    pl.loglog(dts, dts, label = '1 st order', linestyle = '--')
    pl.loglog(dts, [t**2 for t in dts], label = '2 nd order', linestyle = '--')
    pl.legend()
    pl.title('Splitting + time int error test')
    pl.xlabel('dt'); pl.ylabel('Err')
    pl.grid(True, which = 'major')
    if savefig is not None:
        s = f'verify_comb_error_dim_{dim}_ord_{order}.png'
        pl.savefig(savefig  + s, dpi = 100)
    
    pl.figure()
    pl.semilogx(dts, errs['it'], label = 'iterations', marker = 'o')
    pl.legend()
    pl.xlabel('dt'); pl.ylabel('DN iter')
    pl.title('Splitting + time int error test')
    pl.grid(True, which = 'major')
    if savefig is not None:
        s = f'verify_comb_error_iters_dim_{dim}_ord_{order}.png'
        pl.savefig(savefig  + s, dpi = 100)

## same as verify_comb_error, but now also for multirate:
## coarse-coarse, coarse-fine, fine-coarse, C = refinement factor        
def verify_MR_comb(init_cond, tf = 1, k = 10, kmin = 0, theta = None, WR_type = 'DNWR', order = 1, savefig = None, C = 10, TOL = 1e-13, **kwargs):
    assert(type(C) is int)
    if WR_type == 'NNWR':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
        ID_SELF = comm.rank
        
    prob = get_problem(WR_type = WR_type, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    if WR_type == 'NNWR' and ID_SELF == 1:
        pass
    else:
        u1_ref, u2_ref, ug_ref, _ = solve_monolithic(tf, C*2**(k+1), init_cond, order, **kwargs)
        uref = np.hstack([u1_ref, u2_ref, ug_ref])
    if WR_type == 'NNWR': comm.Barrier() ## sync, only one process claculates monolithic sol
    
    if WR_type == 'NNWR' and ID_SELF == 1:
        for n_steps in [2**i for i in range(kmin, k)]:
            ## coarse-coarse
            solver(tf, n_steps, n_steps, init_cond, theta, TOL = TOL)
            ## coarse-fine
            solver(tf, n_steps, C*n_steps, init_cond, theta, TOL = TOL)
            ## fine-coarse
            solver(tf, C*n_steps, n_steps, init_cond, theta, TOL = TOL)
        return None
    
    n, dim = kwargs['n'], kwargs['dim']
    if dim == 1:  L2_fac = np.sqrt(1/(n + 1))
    else:         L2_fac = 1/(n + 1)
    
    dts = []
    errs_cc = {'u': [], 'it': []}
    errs_cf = {'u': [], 'it': []}
    errs_fc = {'u': [], 'it': []}
    for n_steps in [2**i for i in range(kmin, k)]:
        dts.append(tf/n_steps)
        ## coarse-coarse
        u1, u2, ug, E, it = solver(tf, n_steps, n_steps, init_cond, theta, TOL = TOL)
        errs_cc['u'].append(np.linalg.norm(uref - np.hstack([u1, u2, ug]), 2)*L2_fac)
        errs_cc['it'].append(it)
        
        ## coarse-fine
        u1, u2, ug, E, it = solver(tf, n_steps, C*n_steps, init_cond, theta, TOL = TOL)
        errs_cf['u'].append(np.linalg.norm(uref - np.hstack([u1, u2, ug]), 2)*L2_fac)
        errs_cf['it'].append(it)
        
        ## fine-coarse
        u1, u2, ug, E, it = solver(tf, C*n_steps, n_steps, init_cond, theta, TOL = TOL)
        errs_fc['u'].append(np.linalg.norm(uref - np.hstack([u1, u2, ug]), 2)*L2_fac)
        errs_fc['it'].append(it)
        
    pl.figure()
    pl.loglog(dts, errs_cc['u'], label = 'c-c', marker = 'o')
    pl.loglog(dts, errs_cf['u'], label = 'c-f', marker = 'd', linestyle = ':')
    pl.loglog(dts, errs_fc['u'], label = 'f-c', marker = '+')
    pl.loglog(dts, dts, label = '1 st order', linestyle = '--')
    pl.loglog(dts, [t**2 for t in dts], label = '2 nd order', linestyle = '--')
    pl.legend()
    pl.title('MR error test')
    pl.xlabel('dt'); pl.ylabel('Err')
    pl.grid(True, which = 'major')
    if savefig is not None:
        s = f'verify_MR_error_dim_{dim}_ord_{order}_C_{C}.png'
        pl.savefig(savefig  + s, dpi = 100)
        
    pl.figure()
    pl.semilogx(dts, errs_cc['it'], label = 'c-c', marker = 'o')
    pl.semilogx(dts, errs_cf['it'], label = 'c-f', marker = 'd')
    pl.semilogx(dts, errs_fc['it'], label = 'f-c', marker = '+')
    pl.legend()
    pl.xlabel('dt'); pl.ylabel('DN iter')
    pl.title('MR error test')
    pl.grid(True, which = 'major')
    if savefig is not None:
        s = f'verify_MR_error_iters_dim_{dim}_ord_{order}.png'
        pl.savefig(savefig  + s, dpi = 100)
        
    cc_approx_order = [np.log2(errs_cc['u'][i]/errs_cc['u'][i+1]) for i in range(kmin, k-1)]
    cf_approx_order = [np.log2(errs_cf['u'][i]/errs_cf['u'][i+1]) for i in range(kmin, k-1)]
    fc_approx_order = [np.log2(errs_fc['u'][i]/errs_fc['u'][i+1]) for i in range(kmin, k-1)]
    pl.figure()
    pl.title('approximate orders')
    pl.semilogx(dts[:-1], cc_approx_order, label = 'c-c')
    pl.semilogx(dts[:-1], cf_approx_order, label = 'c-f')
    pl.semilogx(dts[:-1], fc_approx_order, label = 'f-c')
    pl.axhline(1, linestyle = '--', label = '1st')
    pl.axhline(2, linestyle = ':', label = '2nd')
    pl.ylim(-1, 3)
    pl.xlabel('dt')
    pl.legend()
    if savefig is not None:
        s = f'verify_MR_approx_ord_dim_{dim}_ord_{order}.png'
        pl.savefig(savefig  + s, dpi = 100)
    
## verify time-itegration order of coupling scheme by comparison with monolithic solution for fixed, small tolerance and increasing number of timesteps    
def verify_test(init_cond, tf = 1, k = 10, kmin = 0, theta = None, WR_type = 'DNWR', order = 1, savefig = None, **kwargs):
    if WR_type == 'NNWR': raise ValueError('no NNWR here')
        
    prob = get_problem(WR_type = WR_type, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    u1_ref, u2_ref, ug_ref, _ = solve_monolithic(tf, 2**(k+1), init_cond, order, **kwargs)
    
    n, dim = kwargs['n'], kwargs['dim']
    if dim == 1:  L2_fac, L2_fac_intf = np.sqrt(1/(n + 1)), 1
    else:         L2_fac, L2_fac_intf = 1/(n + 1), np.sqrt(1/(n + 1))
    
    dts = []
    errs_time = {'u1': [], 'u2': [], 'ug': []}
    errs_split = {'u1': [], 'u2': [], 'ug': []}
    for n_steps in [2**i for i in range(kmin, k)]:
        u1_mono, u2_mono, ug_mono, _ = solve_monolithic(tf, n_steps, init_cond, order, **kwargs)
        errs_time['u1'].append(np.linalg.norm(u1_ref - u1_mono)*L2_fac)
        errs_time['u2'].append(np.linalg.norm(u2_ref - u2_mono)*L2_fac)
        errs_time['ug'].append(np.linalg.norm(ug_ref - ug_mono)*L2_fac_intf)
        
        dts.append(tf/n_steps)
        u1, u2, ug, E, it = solver(tf, n_steps, n_steps, init_cond, theta, TOL = 1e-13)
        errs_split['u1'].append(np.linalg.norm(u1_mono - u1)*L2_fac)
        errs_split['u2'].append(np.linalg.norm(u2_mono - u2)*L2_fac)
        errs_split['ug'].append(np.linalg.norm(ug_mono - ug)*L2_fac_intf)
        
    pl.figure()
    pl.loglog(dts, errs_time['u1'], label = 'u1 time', marker = 'o', linestyle = '-')
    pl.loglog(dts, errs_split['u1'], label = 'u1 split', marker = 'o', linestyle = '-')
    
    pl.loglog(dts, errs_time['u2'], label = 'u2 time', marker = 'd', linestyle = '--')
    pl.loglog(dts, errs_split['u2'], label = 'u2 split', marker = 'd', linestyle = '--')
    
    pl.loglog(dts, errs_time['ug'], label = 'ug time', marker = '+', linestyle = ':')
    pl.loglog(dts, errs_split['ug'], label = 'ug split', marker = '+', linestyle = ':')
    pl.legend()
    pl.title('Splitting + time int error test')
    pl.xlabel('dt'); pl.ylabel('Err')
    pl.grid(True, which = 'major')
    if savefig is not None:
        s = f'testing_dim_{dim}_ord_{order}.png'
        pl.savefig(savefig  + s, dpi = 100)
    
## verify correctness of adaptive coupling scheme via comparison with monolithic (fixed number of steps, many) and decreasing tolerances
def verify_adaptive(init_cond, tf = 1, k_ref = 10, k = 10, WR_type = 'DNWR', which_ref = 'fine', order = -2, savefig = None, **kwargs):
    if WR_type == 'NNWR':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
        ID_SELF = comm.rank
    
    prob = get_problem(WR_type = WR_type, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    if WR_type == 'NNWR' and ID_SELF == 1:
        if which_ref == 'fixed':
            pass
        elif which_ref == 'fine':
            solver(tf, init_cond, TOL = 10**(-k))
    else:
        if which_ref == 'fixed':
            u1_ref, u2_ref, ug_ref, _ = solve_monolithic(tf, 2**k_ref, init_cond, order = order, **kwargs)
        elif which_ref == 'fine':
            u1_ref, u2_ref, ug_ref, _, _, _ = solver(tf, init_cond, TOL = 10**(-k))
        else: raise KeyError('invalid which_ref input, needs to be either fixed or fine')
        
    
    tols = np.array([10**(-i) for i in range(k)])
    
    if WR_type == 'NNWR' and ID_SELF == 1:
        for tol in tols:
            solver(tf, init_cond, TOL = tol)
        return None
    
    n, dim = kwargs['n'], kwargs['dim']
    ## else, also serial
    if dim == 1:  L2_fac, L2_fac_intf = np.sqrt(1/(n + 1)), 1
    else:         L2_fac, L2_fac_intf = 1/(n + 1), np.sqrt(1/(n + 1))
    
    errs = {'u1': [], 'u2': [], 'ug': [], 'it': [], 'steps': []}
    for tol in tols:
        u1, u2, ug, E, it, s = solver(tf, init_cond, TOL = tol)
        errs['u1'].append(np.linalg.norm(u1_ref - u1)*L2_fac)
        errs['u2'].append(np.linalg.norm(u2_ref - u2)*L2_fac)
        errs['ug'].append(np.linalg.norm(ug_ref - ug)*L2_fac_intf)
        errs['it'].append(it)
        errs['steps'].append(s)
    for i in range(k-1):
        print(np.log10(errs['u1'][i]/errs['u1'][i+1]),
              np.log10(errs['u2'][i]/errs['u2'][i+1]),
              np.log10(errs['ug'][i]/errs['ug'][i+1]))
        
    pl.figure()
    pl.loglog(tols, errs['u1'], label = 'u1', marker = 'o')
    pl.loglog(tols, errs['u2'], label = 'u2', marker = 'o')
    pl.loglog(tols, errs['ug'], label = 'ug', marker = 'o')
    pl.loglog(tols, tols, label = '1 st order', linestyle = '--')
    pl.loglog(tols, tols**(1/2), label = 'order 1/2', linestyle = ':')
    pl.legend()
    pl.grid(True, which = 'major')
    pl.title('Adaptive verification')
    pl.xlabel('TOL'); pl.ylabel('Err')
    if savefig is not None:
        s = f'verify_adaptive_dim_{dim}_n_{n}_order_{order}.png'
        pl.savefig(savefig + s, dpi = 100)
    
    pl.figure()
    pl.semilogx(tols, errs['it'], label = 'iterations', marker = 'o')
    pl.legend()
    pl.xlabel('TOL'); pl.ylabel('DN iter')
    pl.grid(True, which = 'major')
    pl.title('Adaptive verification')
    if savefig is not None:
        s = f'verify_adaptive_iters_dim_{dim}_n_{n}_order_{order}.png'
        pl.savefig(savefig + s, dpi = 100)
    
    pl.figure()
    pl.loglog(tols, errs['steps'], label = 'timesteps', marker = 'o')
    pl.loglog(tols, tols**(-1/2), label = 'order 1/2', linestyle = '--')
    pl.loglog(tols, 1/tols, label = 'order 1', linestyle = ':')
    pl.legend()
    pl.xlabel('TOL'); pl.ylabel('steps')
    pl.grid(True, which = 'major')
    pl.title('Adaptive verification')
    if savefig is not None:
        s = f'verify_adaptive_timesteps_dim_{dim}_n_{n}_order_{order}.png'
        pl.savefig(savefig + s, dpi = 100)
    
## simply plot the solution
def visual_verification(init_cond, tf = 1, N_steps = 20, TOL = 1e-14, order = 1, theta = None, WR_type = 'DNWR', savefig = None, len_1 = 1, len_2 = 1, **kwargs):
    if WR_type == 'NNWR':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size != 2: raise ValueError('incorrect number of processes, needs to be exactly 2')
        ID_SELF = comm.rank
    
    prob = get_problem(WR_type = WR_type, len_1 = len_1, len_2 = len_2, **kwargs)
    solver = get_solver(prob, order, WR_type)
    
    if order in [1, 2, 22]: 
        sol = solver(tf, N_steps, N_steps, init_cond, theta, TOL = TOL)
    elif order in [-2, -22]:
        sol = solver(tf, init_cond, TOL = TOL)
    else: raise ValueError('invalid order')
    
    if WR_type == 'NNWR' and ID_SELF == 1: return None # plotting only on first processor
        
    u1, u2, ug = sol[:3]
    
    n, dim = kwargs['n'], kwargs['dim']
    fig = pl.figure()
    if dim == 1:
        res = np.hstack((np.array([0.]), u1, ug, u2, np.array([0.])))
        pl.plot(np.linspace(-len_1, len_2, (len_1 + len_2)*(n+1) + 1), res, label = 'solution')
        pl.axvline(0, color = 'red', linestyle = '--', label = 'interface')
        pl.xlabel('x'); pl.ylabel('u')
        pl.legend()
    elif dim == 2:
        res = np.zeros((n + 2, (len_1 + len_2)*(n+1) + 1))
        res[1:n+1, 1:(len_1  * (n+1) + 1) -1] = np.reshape(u1, (len_1*(n+1) - 1, n)).T
        res[1:n+1, (len_1  * (n+1) + 1) - 1] = ug
        res[1:n+1, -(len_2  * (n+1)):-1] = np.reshape(u2, (len_2*(n+1) - 1, n)).T
        p = pl.pcolor(res)
        fig.colorbar(p)
    else: raise ValueError('invalid dimension')
    
    if savefig is not None:
        s = f'visial_verifiy_{dim}_{n}.png'
        pl.savefig(savefig + s, dpi = 100)