#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:26:44 2020

@author: Peter Meisrimel, Lund University
"""
import numpy as np

def get_dt0(self, tf, TOL, u0, which = 1, k = 2):
    if which == 1:
        a = self.linear_solver(self.M1, self.A1.dot(u0))
        norm = np.sqrt(self.M1.dot(a).dot(a)/self.len_1/self.alpha_1)
    elif which == 2:
        a = self.linear_solver(self.M2, self.A2.dot(u0))
        norm = np.sqrt(self.M2.dot(a).dot(a)/self.len_2/self.alpha_1)
    return (tf*TOL**(1/k))/(100*(1 + norm))
        
def get_new_dt_PI(self, dt, err, TOL, k = 2):
    dt = dt*(TOL/err)**(2/(3*k))*(TOL/self.r_old)**(-1/(3*k))
    self.r_old = err
    return dt

def get_new_dt_deadbeat(self, dt, err, TOL, k = 2):
    return dt*(TOL/err)**(1/k)