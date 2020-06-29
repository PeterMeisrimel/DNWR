# Dirichlet-Neumann (and Neumann Neumann) Waveform Relaxation for coupled heterogeneous heat equations

This is an Python implementation of a Dirichlet-Neumann (and Neumann Neumann) Waveform Relaxation (DNWR/NNWR) solver for coupled heterogeneous heat equations. This features both first and second order multirate, as well as time-adaptive time-integration methods. The code contains 1D and 2D test cases on rectangular domains, using linear finite elements on a equidistant discretization.

## Authors

Peter Meisrimel, Lund University, Sweden, peter.meisrimel@na.lu.se

Azahar Monge, former Lund University

Philipp Birken, Lund University, Sweden

## License

Published under the GNU General Public License v3.0 License

## Software requirements

Python 3.6
mpi4py

## Publications

In preparation

## Related Literature

Monge, Azahar, and Philipp Birken. "On the convergence rate of the Dirichlet–Neumann iteration for unsteady thermal fluid–structure interaction." Computational Mechanics 62.3 (2018): 525-541.

Monge, Azahar, and Philipp Birken. "A Multirate Neumann--Neumann Waveform Relaxation Method for Heterogeneous Coupled Heat Equations." SIAM Journal on Scientific Computing 41.5 (2019): S86-S105.

Monge, Azahar, and Philipp Birken. "A time adaptive Neumann-Neumann waveform relaxation method for thermal fluid-structure interaction."

## Overview

Problem_FSI.py contains the main class describing the discretized heat equation, Problem_FSI_1D.py and Problem_FSI_2D.py are implementations of the corresponding 1D and 2D discretizations.

DNWR_IE.py, DNWR_SDIRK2.py, DNWR_SDIRK2_TA.py and DNWR_SDIRK2_test.py contain the DNWR algorithms based on Implicit Euler, SDIRK2, SDIRK2 + time adaptive and an experimental SDIRK2 variant.
NNWR_IE.py, NNWR_SDIRK2.py, NNWR_SDIRK2_TA.py contain the corresponding versions for NNWR. 

FSI_verification.py contains a wide range of functions for verfication purposes, as well as initial conditions and material parameters. 

Each of the above files contains code to produce verification plots for its associated methods, (Problem_FSI.py for the monolithic (non-coupled) problem). You can run run_all_verify.sh to run all these, to create all verification plots. Note: This may take a very long time, due to many NNWR results not converging and reaching the maximum number of iterations.

verify/code_verification.tex can be used to create the .pdf in showing these verification results in a structured manner. 

The paper_*.py files contain the scripts to create the plots featured in our publication that is in preparation.

## Documentation

Documentation is very minimal. The code is supposed to be read within context of our upcoming publication, since variable naming is aligned with the notation.
Most functions to produce plots and similar are not/barely commented, but work.
