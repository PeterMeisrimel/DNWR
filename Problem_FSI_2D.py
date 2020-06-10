# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:10:47 2019

@author: Azahar Monge, heavy restructuring by Peter Meisrimel
"""
from Problem_FSI import Problem_FSI
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.sparse as sp

class Problem_FSI_2D(Problem_FSI):
    dim = 2
    def __init__(self, n = 10, lambda_1 = 0, lambda_2 = 0, alpha_1 = 0, alpha_2 = 0, WR_type = 'DNWR', len_1 = 1, len_2 = 1):
        super(Problem_FSI_2D, self).__init__(n, lambda_1, lambda_2, alpha_1, alpha_2, WR_type = WR_type, len_1 = len_1, len_2 = len_2)
        self.linear_solver = spsolve
        self.L2_fac = np.sqrt(self.dx)
        self.L2_fac_inner = self.dx
    ## Note: discretization matrices are independent of boundary treatment, i.e. neumann or dirichlet
    ## Matrices only depend on the posisition of the interface
    ## extra neumann flag is to directly assemble system matrix for neumann problem
    ## see Azahar Monges Licentiate or PhD thesis for the details on this discretization
    def compute_matrices(self, nx, ny, alpha, lambda_diff, interface, neumann = False):
        dx = self.dx
        NN = nx*ny
        
        ## correctly set indices for boundary conditions based on their positions
        if interface == 'right':
            intf_idx = -NN + ny
            intf_idx_plus = -1
        elif interface == 'left':
            intf_idx = 0
            intf_idx_plus = 1
        else: raise ValueError('invalid interface position')
        
        Aig = sp.spdiags(-(lambda_diff/(dx**2))*np.ones(NN), intf_idx , NN, ny, format = 'csr')
        Agi = sp.spdiags(-(lambda_diff/(dx**2))*np.ones(NN), -intf_idx, ny, NN, format = 'csr')
        Aggi = sp.spdiags([2*lambda_diff/(dx**2)*np.ones(ny)] + 2*[-0.5 * lambda_diff/(dx**2)*np.ones(ny)], [0, 1, -1], ny, ny, format = 'csr')

        Mig = sp.spdiags([-alpha/12*np.ones(NN), alpha/4 *np.ones(NN)], [intf_idx, intf_idx + intf_idx_plus], NN, ny, format='csr')
        Mgi = sp.spdiags([-alpha/12*np.ones(NN), alpha/4*np.ones(NN)], [-intf_idx, -intf_idx + intf_idx_plus] , ny, NN, format='csr')
        Mggi = sp.spdiags([5*alpha/12*np.ones(ny)] + 2*[-alpha/24*np.ones(ny)], [0, 1, -1], ny, ny, format = 'csr')
        
        B = sp.spdiags(4*np.ones(ny),  0, ny, ny, format = 'csr') - \
            sp.spdiags(  np.ones((2,ny)),  [-1, 1], ny, ny, format = 'csr') 
        Ai = lambda_diff/(dx**2)*(sp.kron(sp.spdiags(np.ones(nx), 0, nx, nx, format = 'csr'), B) + \
             sp.kron(sp.spdiags(np.ones((2,nx)), [-1, 1], nx, nx, format = 'csr'), sp.spdiags(-np.ones(ny), 0, ny, ny, format = 'csr')))
        
        N = sp.spdiags([5/6*np.ones(ny)] + 2*[-1/12*np.ones(ny)], [0, 1, -1], ny, ny, format = 'csr')
        N1 = sp.spdiags([-1/12*np.ones(ny), 1/4*np.ones(ny)], [0, -1], ny, ny, format = 'csr')
        N2 = sp.spdiags([-1/12*np.ones(ny), 1/4*np.ones(ny)], [0, 1], ny, ny, format = 'csr')
        Mi = self.alpha_2*(sp.kron(sp.spdiags(np.ones(nx), 0, nx, nx, format = 'csr'), N) + \
             sp.kron(sp.spdiags(np.ones(nx), -1, nx, nx, format = 'csr'), N1) + \
             sp.kron(sp.spdiags(np.ones(nx), 1, nx, nx, format = 'csr'), N2))
        
        if not neumann:
            return Ai, Mi, Aig, Agi, Aggi, Mig, Mgi, Mggi
        ## else neumann is True
        
        ## also assemble neumann system matrix here
        ## bmat = function for building sparse matrix by blocks
        ## slicing would involve conversions to dense matrices
        N_A = sp.bmat([[Ai, Aig], [Agi, Aggi]], format = 'csr')
        N_M = sp.bmat([[Mi, Mig], [Mgi, Mggi]], format = 'csr')
        
        return Ai, Mi, Aig, Agi, Aggi, Mig, Mgi, Mggi, N_A, N_M
    
    def get_monolithic_matrices(self):
        ## correctly stack up matrices of monolithic system
        ## could also be done with sp.bmat?
        NN1, NN2, ny = self.n_int_1, self.n_int_2, self.n
        NN = NN1 + NN2
        M, A = sp.lil_matrix((NN + ny, NN + ny)), sp.lil_matrix((NN + ny, NN + ny))
        
        M[:NN1, :NN1] = self.M1              ; A[:NN1, :NN1] = self.A1
        M[NN1:-ny, NN1:-ny] = self.M2        ; A[NN1:-ny, NN1:-ny] = self.A2
        M[-ny:, -ny:] = self.Mgg1 + self.Mgg2; A[-ny:, -ny:] = self.Agg1 + self.Agg2
        M[-ny:, :NN1] = self.Mg1             ; A[-ny:, :NN1] = self.Ag1
        M[-ny:, NN1:-ny] = self.Mg2          ; A[-ny:, NN1:-ny] = self.Ag2
        M[:NN1, -ny:] = self.M1g             ; A[:NN1, -ny:] = self.A1g
        M[NN1:-ny, -ny:] = self.M2g          ; A[NN1:-ny, -ny:] = self.A2g
        M, A = M.tocsr(), A.tocsr()
        
        return A, M
    
    def get_initial_values(self, init_cond):
        ## interface assumed to be at x = 0 
        n = self.n
        x1 = np.linspace(-self.len_1, 0, (self.n + 1)*self.len_1 + 1)
        x2 = np.linspace(0, self.len_2, (self.n + 1)*self.len_2 + 1)
        y = np.linspace(0, 1, self.n + 2)
        
        u1, u2, ug = np.zeros(self.n_int_1), np.zeros(self.n_int_2), np.zeros(n)
        for i in range(n): ug[i] = init_cond(0., y[i+1])
        for i in range(n):
            for j, x in enumerate(x1[1:-1]):
                u1[j*n + i] = init_cond(x, y[i+1])
            for j, x in enumerate(x2[1:-1]):
                u2[j*n + i] = init_cond(x, y[i+1])
        return u1, u2, ug