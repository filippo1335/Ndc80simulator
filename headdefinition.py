import numpy as np
nx = 20
pathx = "/Users/filiproch/PycharmProjects/LAMMPS/particle.txt"
# !!!!!!!!!!!
#   DEPRECATED, USE POLYMERDEFINITION
# !!!!!!!!!!!!
def head_definer(path, m):
    coords = np.random.rand(3, m)
    coords[2] = coords[2] * 20 - 10
    coords[1] = 2
    coords[0] = coords[0] * 400 + 300
    coords = coords.T

    with open(path, "w") as f:
        f.write(f'head definition\n\n'
                f'{4*m:10d} atoms\n'
                f'{3*m:10d} bonds\n\n'
                f'{6:10d} atom types\n'
                f'{6:10d} bond types\n\n'
                f'{-300.00:10.4f}{700.00:10.4f} xlo xhi\n'
                f'{-10.00:10.4f}{10.00:10.4f} ylo yhi\n'
                f'{-10.00:10.4f}{10.00:10.4f} zlo zhi\n\n')

        f.write(f'\nAtoms\n\n')
        for i in range(m):
            line = (f"{4*i+1:10d}{2:10d}{i+1:10d}{coords[i, 0]:10.4f}{coords[i, 1]:10.4f}{coords[i, 2]:10.4f}\n"
                    f"{4*i+2:10d}{4:10d}{i+1:10d}{coords[i, 0]+1:10.4f}{coords[i, 1]:10.4f}{coords[i, 2]:10.4f}\n"
                    f"{4*i+3:10d}{5:10d}{i+1:10d}{coords[i, 0]:10.4f}{coords[i, 1]+1:10.4f}{coords[i, 2]:10.4f}\n"
                    f"{4*i+4:10d}{6:10d}{i+1:10d}{coords[i, 0]:10.4f}{coords[i, 1]-1:10.4f}{coords[i, 2]:10.4f}\n")
            f.write(line)

        f.write(f'\nBonds\n\n')
        for i in range(m):
            line = (f"{i+1:10d}{1:10d}{2*i+1:10d}{2*i+2:10d}{""}{""}\n"
                    f"{i+1:10d}{1:10d}{2*i+1:10d}{2*i+2:10d}{""}{""}\n")
            f.write(line)

        f.close()

head_definer(pathx, nx)