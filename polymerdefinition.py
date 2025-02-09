import numpy as np
# Open a file to write the output


def polymerdefiner(path, n, m):
    xcoord = np.arange(2.0, m + 2.0)
    coords = np.random.rand(3, m)
    coords[2] = coords[2] * 15 - 5
    coords[1] = 2
    coords[0] = coords[0] + 10*xcoord + 200
    coords = coords.T
    #coords[(coords > 0) & (coords < 2)] = 2
    #coords[(coords >= 0) & (coords > -2)] = -2
    with open(path, 'w') as f:

        f.write(f'# polymer definition\n\n')
        f.write(f'{2*n + 4*m+1:10d} atoms\n'
                f'{n-1+3*m+10:10d} bonds\n'
                f'{n-2:10d} angles\n'
                f'{n-3:10d} dihedrals\n')

        f.write(f'\n{7:10d} atom types\n'
                f'{7:10d} bond types\n'
                f'{2:10d} angle types\n'
                f'{1:10d} dihedral types\n')

        f.write(f'\n{-10.00:10.4f}{2*n+10:10.4f} xlo xhi'
                f'\n{-50.00:10.4f}{150.00:10.4f} ylo yhi'
                f'\n{-50.00:10.4f}{50.00:10.4f} zlo zhi')

        f.write(f'\n\nMasses\n\n'
                f'{1:10d}{14.02:10.2f}\n'
                f'{2:10d}{5:10.2f}\n'
                f'{3:10d}{1:10.2f}\n'
                f'{4:10d}{1:10.2f}\n'
                f'{5:10d}{1:10.2f}\n'
                f'{6:10d}{1:10.2f}\n'
                f'{7:10d}{1:10.2f}\n')

        f.write(f'\nAtoms\n\n')
        for i in range(1, n + 1):
            line = f"{i:10d}{1:10d}{1:10d}{2*i:10.4f}{0.0000:10.4f}{0.0000:10.4f}\n"
            f.write(line)

        for i in range(n+1, 2*n+1):
            line = f"{i:10d}{1:10d}{3:10d}{2*(i-n):10.4f}{1.0000:10.4f}{0.0000:10.4f}\n"
            f.write(line)

        for i in range(m):
            line = (f"{2*n+4*i+1:10d}{i+2:10d}{2:10d}{coords[i, 0]:10.4f}{coords[i, 1]:10.4f}{coords[i, 2]:10.4f}\n"
                    f"{2*n+4*i+2:10d}{i+2:10d}{4:10d}{coords[i, 0]+1:10.4f}{coords[i, 1]:10.4f}{coords[i, 2]:10.4f}\n"
                    f"{2*n+4*i+3:10d}{i+2:10d}{5:10d}{coords[i, 0]:10.4f}{coords[i, 1]-1:10.4f}{coords[i, 2]:10.4f}\n"
                    f"{2*n+4*i+4:10d}{i+2:10d}{6:10d}{coords[i, 0]:10.4f}{coords[i, 1]+1:10.4f}{coords[i, 2]:10.4f}\n")
            f.write(line)

        f.write(f"{2*n+4*m+1:10d}{1:10d}{7:10d}{280:10.4f}{100:10.4f}{0:10.4f}\n")

        f.write(f'\nBonds\n\n')
        for i in range(1, n):
            line = f"{i:10d}{1:10d}{i:10d}{i+1:10d}{""}{""}\n"
            f.write(line)
        for i in range(m):
            line = (f"{n+3*i:10d}{2:10d}{2*n+4*i+1:10d}{2*n+4*i+2:10d}{""}{""}\n"
                    f"{n+3*i+1:10d}{4:10d}{2*n+4*i+1:10d}{2*n+4*i+3:10d}{""}{""}\n"
                    f"{n+3*i+2:10d}{5:10d}{2*n+4*i+1:10d}{2*n+4*i+4:10d}{""}{""}\n")
            f.write(line)
        for i in range(m):
            line = f"{n+3*m+i+1:10d}{7:10d}{2*n+4*m+1:10d}{2*n+4*i+1:10d}{""}{""}\n"
            f.write(line)

        """        for i in range(n, 2*n):
            line = f"{i:10d}{2:10d}{i+1:10d}{i-n+1:10d}{""}{""}n"
            f.write(line)"""

        f.write(f'\n\nAngles\n\n')
        for i in range(1, n-1):
            line = f"{i:10d}{1:10d}{i:10d}{i+1:10d}{i+2:10d}{""}\n"
            f.write(line)

        f.write(f'\n\nDihedrals\n\n')
        for i in range(1, n-2):
            line = f"{i:10d}{1:10d}{i:10d}{i + 1:10d}{i + 2:10d}{i+3:10d}\n"
            f.write(line)
        f.close()


#polymerdefiner("/Users/filiproch/PycharmProjects/LAMMPS/polymer.txt", 1000, 20)