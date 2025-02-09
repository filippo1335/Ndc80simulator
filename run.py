from lammps import lammps
from plotting import plotter
from polymerdefinition import polymerdefiner
from simdefinition import sim_definer
import os
from mpi4py import MPI
import shutil


"""me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()"""

path = "/Users/filiproch/PycharmProjects/LAMMPS/define_sim_testing.txt"
seedlist = []


heads = 10
polymer = 300

nruns = 1

defaultseed = 40513
steps = 100000
damp = 1.0
temp = 300.0
runName = "verdamp"
graph_dt = 1000
note = "damp 310-360, , no bonding, simplified forcefields"
savesim = True

if f"{runName}" not in os.listdir("figures"): os.mkdir(f"figures/{runName}")
with open(f'figures/{runName}/note.txt', 'w') as f:
    f.write(note)
for i in range(1, nruns+1):
    temp += 10
    new_seed = sim_definer(path, i, defaultseed, steps, damp, temp, True)
    seedlist.append(new_seed)
    polymerdefiner('polymer.txt', polymer, heads)
    lmp = lammps()
    lmp.file('define_sim_testing.txt')
    lmp.close()
    plotter('/Users/filiproch/lammps/dump/particle_coord.txt', heads, True, i, new_seed, steps, graph_dt, runName)
    shutil.copyfile("define_sim_testing.txt", f"figures/{runName}/{i}_{new_seed}.txt")
    if savesim:
        shutil.copyfile(f'/Users/filiproch/lammps/dump/xyz{i}.xyz', f"figures/{runName}/xyz{i}.xyz")
        shutil.copyfile(f'/Users/filiproch/lammps/dump/dcd{i}.dcd', f"figures/{runName}/dcd{i}.dcd")

print(nruns, " runs, with following seeds: ", seedlist)