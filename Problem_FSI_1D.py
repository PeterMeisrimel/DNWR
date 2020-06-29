# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:10:47 2019

@author: Azahar Monge, heavy restructuring by Peter Meisrimel
"""

from Problem_FSI import Problem_FSI
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
    
class Problem_FSI_1D(Problem_FSI):
    dim = 1
    def __init__(self, n = 10, lambda_1 = 0, lambda_2 = 0, alpha_1 = 0, alpha_2 = 0, WR_type = 'DNWR', len_1 = 1, len_2 = 1):
        super(Problem_FSI_1D, self).__init__(n, lambda_1, lambda_2, alpha_1, alpha_2, WR_type = WR_type, len_1 = len_1, len_2 = len_2)
        self.linear_solver = spsolve
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
        
        a_fac = lambda_diff/dx**2
        Aig = sp.lil_matrix((n, 1)); Aig[intf_idx, 0] = -a_fac; Aig.tocsr()
        Agi = sp.lil_matrix((1, n)); Agi[0, intf_idx] = -a_fac; Agi.tocsr()
        Aggi = sp.spdiags(a_fac*np.ones(1), 0 , 1, 1, format = 'csr')
        Ai = sp.spdiags([2*a_fac*np.ones(n), -a_fac*np.ones(n), -a_fac*np.ones(n)], [0, 1, -1], n, n, format = 'csr')
        
        Mig = sp.lil_matrix((n, 1)); Mig[intf_idx, 0] = alpha/6; Mig.tocsr()
        Mgi = sp.lil_matrix((1, n)); Mgi[0, intf_idx] = alpha/6; Mgi.tocsr()
        Mggi = sp.spdiags(alpha/3*np.ones(1), 0, 1, 1, format = 'csr')
        Mi = sp.spdiags([alpha*4/6*np.ones(n), alpha/6*np.ones(n), alpha/6*np.ones(n)], [0, 1, -1], n, n, format = 'csr')
        
        if not neumann:
            return Ai, Mi, Aig, Agi, Aggi, Mig, Mgi, Mggi
        ## else neumann is True
        ## also assemble neumann system matrix then
        N_M = sp.bmat([[Mi, Mig], [Mgi, Mggi]], format = 'csr')
        N_A = sp.bmat([[Ai, Aig], [Agi, Aggi]], format = 'csr')
        
        return Ai, Mi, Aig, Agi, Aggi, Mig, Mgi, Mggi, N_A, N_M
    
    def get_monolithic_matrices(self):
        assert self.WR_type != 'NNWR', 'monolithic matrices are not stored on single processor for NNWR'
        A = sp.bmat([[self.A1, np.zeros((self.n_int_1, self.n_int_2)), self.A1g],
                     [np.zeros((self.n_int_2, self.n_int_1)), self.A2, self.A2g], 
                     [self.Ag1, self.Ag2, self.Agg1 + self.Agg2]], format = 'csr')
        M = sp.bmat([[self.M1, np.zeros((self.n_int_1, self.n_int_2)), self.M1g],
                     [np.zeros((self.n_int_2, self.n_int_1)), self.M2, self.M2g], 
                     [self.Mg1, self.Mg2, self.Mgg1 + self.Mgg2]], format = 'csr')
        return A, M
    
    def get_initial_values(self, init_cond):
        x1 = np.linspace(-self.len_1, 0, (self.n + 1)*self.len_1 + 1)
        u1 = np.array([init_cond(x) for x in x1])
        
        x2 = np.linspace(0, self.len_2, (self.n + 1)*self.len_2 + 1)
        u2 = np.array([init_cond(x) for x in x2])
        
        ug = np.array([init_cond(0)])
        return u1[1:-1], u2[1:-1], ug
    
if __name__ == '__main__':
    from FSI_verification import get_parameters
    ## discrete L2 norm test
    pp = get_parameters('test')
    n_list = [2**i for i in range(12)]
    discr, mass = [], []
    for n in n_list:
        prob = Problem_FSI_1D(n, **pp, len_1 = 3, len_2 = 4)
        u0_f = lambda x: 500
        
        u1, u2, ug = prob.get_initial_values(u0_f)
        uu = np.hstack((u1, u2, ug))
        print(prob.norm_interface(ug),
              prob.norm_inner(u1, 'D'),
              prob.norm_inner(np.hstack([u2, ug]), 'N'),
              prob.norm_inner(u1, u2, ug))