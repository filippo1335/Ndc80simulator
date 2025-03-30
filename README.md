# Ndc80 simulator

## sim_numba.ipynb is intended for running the sims

sim.ipynb is legacy code in pure numpy

sim_numba.ipynb compiles in numba using njit for better performace, however it is less readable since most of the multidimensional math operations were converted to 1D vector operations.

sim_numba.ipynb also contains a cell with ipyparallel code allowing for running multiple simulations in parallel with variable paramemters. For best performance n_proc = n_cores.

plotter() is self explanatory

# Installation guide

```
git clone https://github.com/filippo1335/Ndc80simulator.git


```