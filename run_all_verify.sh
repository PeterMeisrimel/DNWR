python3 create_folders.py
## running this takes a long time, mostly because of the NNWR not properly converging
python3 Problem_FSI.py
python3 DNWR_IE.py
python3 DNWR_SDIRK2.py
python3 DNWR_SDIRK2_test.py
python3 DNWR_SDIRK2_TA.py
mpiexec -n 2 python3 NNWR_IE.py
mpiexec -n 2 python3 NNWR_SDIRK2.py
mpiexec -n 2 python3 NNWR_SDIRK2_TA.py