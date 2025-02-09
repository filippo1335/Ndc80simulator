import numpy as np

pathx = "/Users/filiproch/PycharmProjects/LAMMPS/define_sim_testing.txt"
seedx = np.random.randint(1, 99999)


def sim_definer(path, nrun, seed, steps, damp, temp, reseed):

    if reseed:
        seed = np.random.randint(1, 99999)

    text = f"""
    # Initialization
    units		real
    boundary	p p p
    atom_style	molecular
    read_data 	/Users/filiproch/PycharmProjects/LAMMPS/polymer.txt extra/special/per/atom 10

    neighbor	1 bin
    neigh_modify every 10 one 100

    bond_style  hybrid harmonic
    bond_coeff  * harmonic 0 1
    bond_coeff	1 harmonic 5 2.0
    bond_coeff	2 harmonic 5 1.0
    bond_coeff  3 harmonic 200 0.2
    bond_coeff  6 harmonic 200 0.2
    bond_coeff  7 harmonic 0.1 105

    angle_style     harmonic
    angle_coeff	1 100 180
    angle_coeff 2 20 180

    dihedral_style	zero
    dihedral_coeff	*


    pair_style	lj/cut 10.5
    pair_coeff * * 0 1.0 2.5
    #pair_coeff	1 1 0 1 10
    pair_coeff 	1 2 5 3.5 40 #polymer vs head
    #pair_coeff  1 3 0 1.1 3
    #pair_coeff  1 4 0 1.2 2
    pair_coeff	2 2 1 2.1 10
    #pair_coeff  2 3 0 2 2
    #pair_coeff  2 4 0 1 1
    pair_coeff  3 3 0.1 2.2 10
    pair_coeff  3 4 20 0.2 15
    #pair_coeff  4 4 0 1 1
    pair_coeff 5 6 0.2 0.2 15

    group POLYMER type 1 3
    group PART type 2 4 5 6
    group BEAD type 2
    group SURFACE type 3 4




    dump 		1 all xyz 10000000 /Users/filiproch/lammps/dump/xyz{nrun}.xyz
    dump 		2 all dcd 100 /Users/filiproch/lammps/dump/dcd{nrun}.dcd
    dump		3 BEAD xyz 1000 /Users/filiproch/lammps/dump/particle_coord.txt
    #dump        4 PART custom 10 /Users/filiproch/lammps/dump/custom.txt x


    #####################################################
    # Equilibration (Langevin dynamics at 300 K)


    velocity all create {temp} {seed}
    run 0 # temperature may not be 300K
    velocity all scale {temp} # now it should be

    fix     2 PART bond/create/angle 1 3 4 6 3 aconstrain 120 180 prob 0.5 {seed} iparam 1 3 jparam 1 4
    fix     3 PART bond/break 1 3 0.4 prob 0.001 {seed}
    fix     5 PART bond/create/angle 1 5 6 6 6 aconstrain 120 180 prob 0.5 {seed} iparam 1 5 jparam 1 6
    fix     6 PART bond/break 1 6 0.4 prob 0.001 {seed}
    fix		4 PART rigid/nve/small molecule langevin {temp} {temp} {damp} {seed} #treats beads as one rigid body
    thermo_style	custom step temp
    thermo          100000
    timestep	1
    run		{steps}
    """

    with open(path, "w") as f:
        f.write(text)
        f.close()
    print(seed)
    return seed

    # print(text)
