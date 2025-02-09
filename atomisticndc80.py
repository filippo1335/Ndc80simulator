from lammps import lammps
import numpy as np
import os

text = f""

with open("ndc80sim.txt", "w") as f:
        f.write(text)
        f.close()


lmp = lammps()
lmp.file('ndc80sim.txt')
lmp.close()