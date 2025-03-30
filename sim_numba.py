import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time
import os


fastmath = False
cache = True

angles = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                   [np.pi/2, np.pi/2, 4*np.pi/9, np.pi/9, 4*np.pi/9],
                   [np.pi, 17*np.pi/18, 5*np.pi/9, 5*np.pi/9, 17*np.pi/18],
                   [5*np.pi/9, 17*np.pi/18, np.pi, 17*np.pi/18, 5*np.pi/9],
                   [4*np.pi/9, np.pi/2, np.pi/2, 4*np.pi/9, np.pi/9]])


#-----------------------------------------------------------
# SYSTEM CREATION
#-----------------------------------------------------------
@njit(fastmath=fastmath, cache=cache)
def polymergenerator(polymerl, headl, unit_distance= 4.13):      # 4.13 nm
    # Center-of-mass (COM)
    COM = np.array([16.8, 38.0, 20.1])
    # Create a linear polymer: each row is a 3D coordinate (x, y, z)
    polymer = np.empty((polymerl, 3))
    for i in range(polymerl):
        polymer[i, 0] = float(i) *unit_distance
        polymer[i, 1] = 20.0
        polymer[i, 2] = 20.0
    # Create polymer surface as 2 x 14 copies of the polymer, with offsets on the circumference of the circle with radius = 14.875 nm
    #                           2 x because of two binding sites
    num_copies = 14
    polymersurfaceP = np.empty((polymerl * num_copies, 3))
    polymersurfaceN = np.empty((polymerl * num_copies, 3))
    for k in range(num_copies):
        angle = 2.0 * np.pi * k / num_copies
        offset0 = 0.0
        offset1 = 14.875 * np.cos(angle) #final radius = 14.875 nm
        offset2 = 14.875 * np.sin(angle)
        for i in range(polymerl):
            polymersurfaceP[k * polymerl + i, 0] = polymer[i, 0] + offset0
            polymersurfaceP[k * polymerl + i, 1] = polymer[i, 1] + offset1
            polymersurfaceP[k * polymerl + i, 2] = polymer[i, 2] + offset2
            polymersurfaceN[k * polymerl + i, 0] = polymersurfaceP[k * polymerl + i, 0] + 0.1389 *unit_distance
            polymersurfaceN[k * polymerl + i, 1] = polymersurfaceP[k * polymerl + i, 1] + 0.0
            polymersurfaceN[k * polymerl + i, 2] = polymersurfaceP[k * polymerl + i, 2] + 0.0

    # create offsets for the ndc80 head surface
    offsets = np.array([[0.4, 0.0, 0.0],
                        [0.0, 0.4, 0.0],
                        [-0.4, 0.0, 0.0],
                        [-0.0694, -0.3939, 0.0],
                        [0.0694, -0.3939, 0.0]])
    headsurface = COM + offsets * unit_distance
    return polymer, COM, headsurface, polymersurfaceP, polymersurfaceN


#-----------------------------------------------------------
# FORCES
#-----------------------------------------------------------
@njit(fastmath=fastmath, cache=cache)
def lj(a, b, epsilon, sigma): #0.001, 0.2
    f_total = np.zeros(3)
    for i in range(a.shape[0]):
        dif0 = a[i, 0] - b[0]
        dif1 = a[i, 1] - b[1]
        dif2 = a[i, 2] - b[2]
        dist = np.sqrt(dif0 * dif0 + dif1 * dif1 + dif2 * dif2)
        #if sigma/dist>0.1:
        #    dist = sigma/10
        f_val = 24 * epsilon * (-2 * (sigma ** 12 / (dist ** 13)) + (sigma ** 6 / (dist ** 7)))
        inv_dist = 1.0 / dist
        f_total[0] += f_val * dif0 * inv_dist
        f_total[1] += f_val * dif1 * inv_dist
        f_total[2] += f_val * dif2 * inv_dist
    return f_total


@njit(fastmath=fastmath, cache=cache)
def lj_repulsive(a, b, epsilon=0.1, sigma=1.0):
    f_total = np.zeros(3)
    sigma_inv = 1.0 / sigma
    for i in range(a.shape[0]):
        #dif0 = 0 ##a[i, 0] - b[0]  test with 0.0 to effectively make the curface a cylinder; if calculating diff -> balls
        dif1 = a[i, 1] - b[1]
        dif2 = a[i, 2] - b[2]
        dist = np.sqrt(dif1 * dif1 + dif2 * dif2)
        if dist*sigma_inv > 1.13:
            continue
        f_val = -48 * epsilon * (sigma ** 12 / (dist ** 13))
        inv_dist = 1.0 / dist
        #f_total[0] += f_val * dif0 * inv_dist
        f_total[1] += f_val * dif1 * inv_dist
        f_total[2] += f_val * dif2 * inv_dist
    return f_total


@njit(fastmath=fastmath, cache=cache)
def harmonic_angle(surface, COM, n, angles, K=0.00001):
    N = surface.shape[0]
    force = np.empty((N, 3))
    for i in range(N):
        j = (i + n) % N
        # Compute vectors from COM to surface atoms
        ab0 = surface[i, 0] - COM[0]
        ab1 = surface[i, 1] - COM[1]
        ab2 = surface[i, 2] - COM[2]
        bc0 = surface[j, 0] - COM[0]
        bc1 = surface[j, 1] - COM[1]
        bc2 = surface[j, 2] - COM[2]
        ab_norm = np.sqrt(ab0 * ab0 + ab1 * ab1 + ab2 * ab2)
        bc_norm = np.sqrt(bc0 * bc0 + bc1 * bc1 + bc2 * bc2)
        if ab_norm == 0.0 or bc_norm == 0.0:
            force[i, 0] = 0.0
            force[i, 1] = 0.0
            force[i, 2] = 0.0
            continue
        cos_angle = (ab0 * bc0 + ab1 * bc1 + ab2 * bc2) / (ab_norm * bc_norm)
        if cos_angle > 1.0:
            cos_angle = 1.0
        elif cos_angle < -1.0:
            cos_angle = -1.0
        angle_diff = angles[n,i] - np.arccos(cos_angle)
        # Compute cross products for the double cross product force
        cross_inner0 = bc1 * ab2 - bc2 * ab1
        cross_inner1 = bc2 * ab0 - bc0 * ab2
        cross_inner2 = bc0 * ab1 - bc1 * ab0
        vec0 = ab1 * cross_inner2 - ab2 * cross_inner1
        vec1 = ab2 * cross_inner0 - ab0 * cross_inner2
        vec2 = ab0 * cross_inner1 - ab1 * cross_inner0
        vec_norm = np.sqrt(vec0 * vec0 + vec1 * vec1 + vec2 * vec2)
        if vec_norm != 0.0:
            factor = -2 * K * angle_diff / vec_norm
            force[i, 0] = factor * vec0
            force[i, 1] = factor * vec1
            force[i, 2] = factor * vec2
        else:
            force[i, 0] = 0.0
            force[i, 1] = 0.0
            force[i, 2] = 0.0
    return force


@njit(fastmath=fastmath, cache=cache)
def harmonic_surface(surface, COM, epsilon=0.00001, sigma=1.652): # 0.1 GJ/mol , 1.652 nm
    N = surface.shape[0]
    force = np.empty((N, 3))
    for i in range(N):
        dx = surface[i, 0] - COM[0]
        dy = surface[i, 1] - COM[1]
        dz = surface[i, 2] - COM[2]
        norm_val = np.sqrt(dx * dx + dy * dy + dz * dz)
        if norm_val > 0:
            factor = 2 * epsilon * (sigma - norm_val) / norm_val
            force[i, 0] = factor * dx
            force[i, 1] = factor * dy
            force[i, 2] = factor * dz
        else:
            force[i, 0] = 0.0
            force[i, 1] = 0.0
            force[i, 2] = 0.0
    return force


#-----------------------------------------------------------
# TIME INTEGRATION
#-----------------------------------------------------------
@njit(fastmath=fastmath, cache=cache)
def notverlet(COM, surface, dt, random_val, mu, polymer, polymersurfaceP, polymersurfaceN, epsi):
    # Compute harmonic angle forces and sum them
    # N = surface.shape[0]
    force1 = harmonic_angle(surface, COM, 1, angles)
    force2 = harmonic_angle(surface, COM, 2, angles)
    force3 = harmonic_angle(surface, COM, -2, angles)
    force4 = harmonic_angle(surface, COM, -1, angles)
    angle_forces = force1 + force2 + force3 + force4
    # Compute harmonic bond force
    harmonic_bond = harmonic_surface(surface, COM)

    # Compute Lennard-Jones (Brownian) force on a particular surface atom (index 3)
    lj_force_surf = np.zeros_like(surface)
    lj_force_surf[3] = lj(polymersurfaceN, surface[3], epsi, 0.14) # 0.00413  0.2*4.13 = 0.82 GJ/mol nm  0.00244613
    lj_force_surf[4] = lj(polymersurfaceP, surface[4], epsi, 0.14) # 0.00413 0.2*4.13 = 0.826  0.00244613
    delta_surface = np.empty_like(surface)
    # Update surface positions
    for i in range(surface.shape[0]):
        for j in range(3):
            delta_surface[i,j] = dt * mu *(angle_forces[i, j] + harmonic_bond[i, j] + lj_force_surf[i, j]) #+ 0.001*np.random.uniform(-random_val, random_val) )
        #displacement = np.sqrt(delta_surface[i,0]**2 + delta_surface[i,1]**2 + delta_surface[i,2]**2)
        #if displacement > 0.1: delta_surface= 0.1*delta_surface/displacement
    new_surface = np.empty_like(surface)
    for i in range(surface.shape[0]):
        for j in range(3):
            new_surface[i, j] = (surface[i, j] + delta_surface[i,j])
                                 #dt * ((angle_forces[i, j] + harmonic_bond[i, j] + lj_force_surf[i, j]) + 0.1*np.random.uniform(-random_val, random_val) ))
    # Compute repulsive force on COM from polymer
    lj_force_COM = lj_repulsive(polymer, COM, 0.00005, 14.875) # 0.00413 14.8(MT radius) + 1.6(head radius)
    # Sum harmonic bond forces over surface atoms (exclude angle forces per corrected dynamics)
    harmonic_sum = np.zeros(3)
    for i in range(harmonic_bond.shape[0]):
        harmonic_sum[0] += harmonic_bond[i, 0]
        harmonic_sum[1] += harmonic_bond[i, 1]
        harmonic_sum[2] += harmonic_bond[i, 2]

    # Update COM position
    delta_COM = np.empty(3)
    for i in range(3):
        delta_COM[i] = dt * (mu * (lj_force_COM[i] - harmonic_sum[i]) + np.random.uniform(-random_val, random_val)) #mu is in time units
    #displacement = np.sqrt(delta_COM[0]**2+delta_COM[1]**2+delta_COM[2]**2)
    #if displacement > 0.1: delta_COM = 0.1*delta_COM/displacement
    new_COM = np.empty(3)
    for i in range(3):
        new_COM[i] = COM[i] + delta_COM[i]
    # limit the box, if moves to far, get pushed back; in the future: make a counter for how many times it crossed x-axis boundary to use for analytics - virtually extends the x-axis
    if new_COM[0]<4.13:
        new_COM[0] += 12.39
        new_surface[:,0] += 12.39
    elif new_COM[0]>33.04:
        new_COM[0] -= 12.39
        new_surface[:,0] -= 12.39
    if new_COM[1]>100.0:
        new_COM[1] -= 50.0
        new_surface[:,1] -= 50.0
    elif new_COM[1]< -60.0:
        new_COM[1] += 50.0
        new_surface[:,1] += 50.0
    if new_COM[2]>100.0:
        new_COM[2] -= 50.0
        new_surface[:,2] -= 50.0
    elif new_COM[2]< -60.0:
        new_COM[2] += 50.0
        new_surface[:,2] += 50.0
    return new_COM, new_surface


#-----------------------------------------------------------
# SIMULATE
#-----------------------------------------------------------
@njit(fastmath=fastmath, cache=cache)
def simulate(COM, surface, polymer, polymersurfaceP, polymersurfaceN, steps, dt, random_val, mu, print_every, hist_every=1000, epsi=0.001244613):
    histsize = steps // hist_every
    headhist = np.empty((histsize, 3))
    surfacehist = np.empty((histsize, surface.shape[0], 3))
    headhist[0] = COM
    surfacehist[0] = surface
    for i in range(steps + 1):
        COM, surface = notverlet(COM, surface, dt, random_val, mu, polymer, polymersurfaceP, polymersurfaceN, epsi)
        if i % hist_every == 0:
            headhist[i // hist_every] = COM
            surfacehist[i // hist_every] = surface
        if i % print_every == 0:
            print("Step", i, "/", steps, "COM:", COM[0], COM[1], COM[2], flush=True)
            #print(surface)
    return headhist, surfacehist


def plotter(headhistv,id, save=False):
    plt.style.use('dark_background')
    render_everyN = 1
    orthogonal_distance = np.linalg.norm(headhistv[::render_everyN, 1:3]-[20,20], axis=1)
    time_arr = np.arange(steps/(render_everyN*hist_every))*render_everyN*hist_every*dt/(steps *10^9)

    #fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [4, 4]}, figsize=(10, 10))
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(10,10), dpi=200, gridspec_kw={'height_ratios': [3, 1,2]}) #gridspec_kw={'height_ratios': [4, 1], "width_ratios": [1]}
    pos = ax[0,0].scatter(headhistv[::render_everyN, 2]-20, headhistv[::render_everyN, 1]-20, cmap="gist_rainbow", c=time_arr, s=0.1)
    ax[0,0].set_xlim(-25, 25)
    ax[0,0].set_ylim(-25, 25)
    ax[0,0].set_title("MT crosssection (y and z coordinates) over time", fontsize=8)
    ax[0,0].set_xlabel("z coordinate (nm)", fontsize=8)
    ax[0,0].set_ylabel("y coordinate (nm)", fontsize=8)
    ax[0,0].set_aspect("equal", adjustable="box")
    perp = ax[0,1].scatter(headhistv[::render_everyN, 0], orthogonal_distance[::], cmap="gist_rainbow", c=time_arr, s=0.1)
    ax[0,1].set_xlim(0, 40)
    ax[0,1].set_ylim(0, 40)
    ax[0,1].set_title("orthogonal distance from the central axis of the MT and x-coordinate (along MT) over time", fontsize=8)
    ax[0,1].set_xlabel("x coordinate (nm)", fontsize=8)
    ax[0,1].set_ylabel("orthogonal distance (nm)", fontsize=8)
    ax[0,1].set_aspect("equal", adjustable="box")
    ax[2,1].hist(orthogonal_distance, bins=100, range=(16.0,18.5))
    ax[2,0].hist(orthogonal_distance[1:]-orthogonal_distance[:-1], bins=90)
    gs = ax[1, 0].get_gridspec()
    fig.colorbar(pos, ax=ax[0,:], label='time (µs)', orientation='horizontal', shrink=0.95)
    # remove the underlying Axes
    for ax in ax[1,:]:
        ax.remove()
    axbig = fig.add_subplot(gs[1, :])
    figigi = axbig.plot(time_arr[::], orthogonal_distance[::], linewidth=1)
    axbig.set_ylim(10,100)
    axbig.set_title("orthogonal distance from the center over time", fontsize=8)
    axbig.set_xlabel('time (µs)')
    axbig.set_ylabel('orthogonal distance (nm)')
    plt.show()
    if save:
        fig.savefig(f'/data/home/bt23708/Ndc80sim/engine{id}.png')
    print(f"orthogonal_diff_mean:{(orthogonal_distance[1:]-orthogonal_distance[:-1]).mean()}")



#-----------------------------------------------------------
# SINGLE CORE SIMS
#-----------------------------------------------------------
polymerLength = 10 # ten repeats
headN = 1 # one head
steps = 100000000
dt = 10 #fs
#random_val = 0.01 # nm/sqrt(fs)
mu = 0.01 #fs mol/g, gives the viscosity of the medium ~ 5.34 mPa*s
print_every = 1000000
hist_every = 1000
unit_distanceX = 4.13 #nm
temp = 310 #K
random_val = np.sqrt(2.0*8.314*(10.0**-3) * mu * temp)/1000.0



# RUN THE SIM
polymer, COM, surface, polymersurfaceP, polymersurfaceN = polymergenerator(polymerLength, headN)
start = time.perf_counter()
headhist, surfacehist = simulate(COM, surface, polymer, polymersurfaceP, polymersurfaceN , steps, dt, random_val, mu, print_every, hist_every, epsi=0.00001)
print("Simulation complete. Final COM:", headhist[-1], flush=True)
end = time.perf_counter()
print("Elapsed (after compilation) = {}s".format((end - start)), flush=True)
plotter(headhist, 0, True)
print("finished", flush=True)
