#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:16:08 2020

@author: Peter Meisrimel, Lund University
"""
from numpy import sin, cos, pi

#############################################################
#  DNWR Relaxation
#############################################################
## see NNWR paper formulas (9.2) (9.3)
## Monge, Azahar, and Philipp Birken.
## "A Multirate Neumann--Neumann Waveform Relaxation Method for Heterogeneous Coupled Heat Equations."
## SIAM Journal on Scientific Computing 41.5 (2019): S86-S105.
def w_i(self, a, lam, dt):
    dx = self.dx
    return sum([3*dt*dx**2 * sin(i*pi*dx)**2/(2*a*dx**2 + 6*lam*dt + (a*dx**2 - 6*lam*dt) * cos(i*pi*dx)) for i in range(1, self.n + 1)])
def S_i(self, a, lam, dt):
    dx = self.dx
    return (6*dt*dx*(a*dx**2 + 3*lam*dt) - (a*dx**2 - 6*lam*dt)**2 * self.w_i(a, lam, dt))/(18*dt**2*dx**3)

def theoretical_conv_rate(self, dt, theta):
    S1 = self.S_i(self.alpha_1, self.lambda_1, dt)
    S2 = self.S_i(self.alpha_2, self.lambda_2, dt)
    return abs(1 - theta*(1 + S1/S2))
        
def DNWR_theta_opt(self, dt1, dt2):
    dt = max(dt1, dt2)
    S1 = self.S_i(self.alpha_1, self.lambda_1, dt)
    S2 = self.S_i(self.alpha_2, self.lambda_2, dt)
    return abs(1/(1 + S1/S2))
    
## time adaptive case, see DNWR paper
def DNWR_theta_opt_TA(self, t1, t2):
    # maximum of average step-size
    dt = max(t1[-1]/(len(t1) - 1), t2[-1]/(len(t2) - 1))
    return self.DNWR_theta_opt(dt, dt)

#############################################################
#  NNWR: Relaxation
#############################################################
def NNWR_theta_opt(self, dt1, dt2):
    dt = max(dt1, dt2)
    S1 = self.S_i(self.alpha_1, self.lambda_1, dt)
    S2 = self.S_i(self.alpha_2, self.lambda_2, dt)
    return abs(1/(2 + S2/S1 + S1/S2))
    
def NNWR_theta_opt_TA(self, t1, t2):
    dt1, dt2 = t1[-1]/(len(t1) - 1), t2[-1]/(len(t2) - 1)
    return self.NNWR_theta_opt(dt1, dt2)

def theoretical_conv_rate_NNWR(self, dt, theta):
    S1 = self.S_i(self.alpha_1, self.lambda_1, dt)
    S2 = self.S_i(self.alpha_2, self.lambda_2, dt)
    return abs(1 - theta*(2 + S1/S2 + S2/S1))

############
# testing
############
def DNWR_theta_opt_test(self, dt1, dt2):
    S1 = self.S_i(self.alpha_1, self.lambda_1, dt1)
    S2 = self.S_i(self.alpha_2, self.lambda_2, dt2)
    return abs(1/(1 + S1/S2))

def DNWR_theta_opt_test_TA(self, t1, t2):
    # maximum of average step-size
    return self.DNWR_theta_opt_test(t1[-1]/(len(t1) - 1), t2[-1]/(len(t2) - 1))

def NNWR_theta_opt_test(self, dt1, dt2):
    S1 = self.S_i(self.alpha_1, self.lambda_1, dt1)
    S2 = self.S_i(self.alpha_2, self.lambda_2, dt2)
    return abs(1/(2 + S2/S1 + S1/S2))
    
def NNWR_theta_opt_test_TA(self, t1, t2):
    dt1, dt2 = t1[-1]/(len(t1) - 1), t2[-1]/(len(t2) - 1)
    return self.NNWR_theta_opt_test(dt1, dt2)