#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:26:44 2020

@author: Peter Meisrimel, Lund University
"""

import numpy as np

def get_new_dt(self, dt, err, TOL):
    err_ind = np.sqrt((1/len(err))*((err/(TOL*self.norm_L2(err, True) + TOL))**2))
    err_ind_norm = self.norm_L2(err_ind, True)
    if err_ind_norm > 1:
        return dt*max(0.5, 0.9*err_ind_norm**(-1/2))
    else:
        return dt*min(2, 0.9*err_ind_norm**(-1/2))
    
def get_dt0(self, tf, TOL, u0, which = 1, k = 2):
    if which == 1:
        a = self.linear_solver(self.M1, self.A1.dot(u0))
    elif which == 2:
        a = self.linear_solver(self.M2, self.A2.dot(u0))
    return (tf*TOL**(1/k))/(100*(1 + self.norm_L2(a, True)))
        
def get_new_dt_PI(self, dt, err, TOL, k = 2):
    r = self.norm_L2(err, True)
    dt = dt*(TOL/r)**(2/(3*k))*(TOL/self.r_old)**(-1/(3*k))
    self.r_old = r
    return dt

def get_new_dt_deadbeat(self, dt, err, TOL, k = 2):
    return dt*(TOL/self.norm_L2(err, True))**(1/k)