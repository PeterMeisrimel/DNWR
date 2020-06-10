# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:10:47 2019

@author: Azahar Monge, heavy restructuring by Peter Meisrimel
"""

from Problem_FSI import Problem_FSI
import numpy as np
    
class Problem_FSI_1D(Problem_FSI):
    dim = 1
    def __init__(self, n = 10, lambda_1 = 0, lambda_2 = 0, alpha_1 = 0, alpha_2 = 0, WR_type = 'DNWR', len_1 = 1, len_2 = 1):
        super(Problem_FSI_1D, self).__init__(n, lambda_1, lambda_2, alpha_1, alpha_2, WR_type = WR_type, len_1 = len_1, len_2 = len_2)
        self.linear_solver = np.linalg.solve
        self.L2_fac = 1.
        self.L2_fac_inner = np.sqrt(self.dx)
    ## Note: discretization matrices are independent of boundary treatment, i.e. neumann or dirichlet
    ## Matrices only depend on the posisition of the interface
    ## extra neumann flag is to directly assemble system matrix for neumann problem
    ## see Azahar Monges Licentiate or PhD thesis for the details on this discretization
    def compute_matrices(self, nx, ny, alpha, lambda_diff, interface, neumann = False):
        dx = self.dx
        n = nx
        
        ## correctly set indices for boundary conditions based on their positions
        if interface == 'right':
            intf_idx = -1
        elif interface == 'left':
            intf_idx = 0
        else: raise ValueError('invalid interface position')
        
        Aig = np.zeros((n, 1)); Aig[intf_idx, 0] = -lambda_diff/(dx**2)
        Agi = np.zeros((1, n)); Agi[0, intf_idx] = -lambda_diff/(dx**2)
        Aggi = np.array([[lambda_diff/(dx**2)]])
    
        Mig = np.zeros((n, 1)); Mig[intf_idx, 0] = alpha/6
        Mgi = np.zeros((1, n)); Mgi[0, intf_idx] = alpha/6
        Mggi = np.array([[alpha/3]])
        
        Ai = np.diag(n*[2*lambda_diff/(dx**2)]) + np.diag((n-1)*[-(lambda_diff/(dx**2))],-1) + np.diag((n-1)*[-(lambda_diff/(dx**2))],1)
        Mi = np.diag(n*[4*alpha/6]) + np.diag((n-1)*[alpha/6],-1) + np.diag((n-1)*[alpha/6],1)
        
        if not neumann:
            return Ai, Mi, Aig, Agi, Aggi, Mig, Mgi, Mggi
        ## else neumann is True
        ## also assemble neumann system matrix then
        N_A = np.zeros((n+1,n+1)); N_M = np.zeros((n+1,n+1))
        N_A[:n,:n] = Ai  ; N_M[:n,:n] = Mi
        N_A[n:,:n] = Agi ; N_M[n:,:n] = Mgi
        N_A[:n,n:] = Aig ; N_M[:n,n:] = Mig
        N_A[n:,n:] = Aggi; N_M[n:,n:] = Mggi
        
        return Ai, Mi, Aig, Agi, Aggi, Mig, Mgi, Mggi, N_A, N_M
    
    def get_monolithic_matrices(self):
        n1, n2 = self.n_int_1, self.n_int_2
        NN = n1 + n2
        
        M, A = np.zeros((NN + 1, NN + 1)), np.zeros((NN + 1, NN + 1))
        M[:n1, :n1] = self.M1              ; A[:n1, :n1] = self.A1
        M[n1:NN, n1:NN] = self.M2          ; A[n1:NN, n1:NN] = self.A2
        M[NN:, NN:] = self.Mgg1 + self.Mgg2; A[NN:, NN:] = self.Agg1 + self.Agg2
        M[NN:, :n1] = self.Mg1             ; A[NN:, :n1] = self.Ag1
        M[NN:, n1:NN] = self.Mg2           ; A[NN:, n1:NN] = self.Ag2
        M[:n1, NN:] = self.M1g             ; A[:n1, NN:] = self.A1g
        M[n1:NN, NN:] = self.M2g           ; A[n1:NN, NN:] = self.A2g
        return A, M
    
    def get_initial_values(self, init_cond):
        x1 = np.linspace(-self.len_1, 0, (self.n + 1)*self.len_1 + 1)
        u1 = np.array([init_cond(x) for x in x1])
        
        x2 = np.linspace(0, self.len_2, (self.n + 1)*self.len_2 + 1)
        u2 = np.array([init_cond(x) for x in x2])
        
        ug = np.array([init_cond(0)])
        return u1[1:-1], u2[1:-1], ug