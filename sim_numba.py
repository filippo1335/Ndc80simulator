import math

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time
import os
import tracemalloc
import sys


fastmath = False
cache = False

angles = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                   [np.pi/2, np.pi/2, 4*np.pi/9, np.pi/9, 4*np.pi/9],
                   [np.pi, 17*np.pi/18, 5*np.pi/9, 5*np.pi/9, 17*np.pi/18],
                   [5*np.pi/9, 17*np.pi/18, np.pi, 17*np.pi/18, 5*np.pi/9],
                   [4*np.pi/9, np.pi/2, np.pi/2, 4*np.pi/9, np.pi/9]])


#-----------------------------------------------------------
# SYSTEM CREATION
#-----------------------------------------------------------
#@njit(fastmath=fastmath, cache=cache)
def polymergenerator(polymerl, headl, unit_distance=4.13):
    # Create polymer: each row is a 3D coordinate (x,y,z)
    polymer = np.empty((polymerl, 3))
    for i in range(polymerl):
        polymer[i, 0] = float(i) * unit_distance
        polymer[i, 1] = 20.0
        polymer[i, 2] = 20.0

    # Create polymer surface as 2 x 14 copies of the polymer, with offsets on a circle (radius = 14.875 nm)
    num_copies = 14
    polymersurfaceP = np.empty((num_copies, polymerl, 3))
    polymersurfaceN = np.empty((num_copies, polymerl, 3))
    for k in range(num_copies):
        angle = 2.0 * np.pi * k / num_copies
        offset0 = 0.0
        offset1 = 14.875 * np.cos(angle) #final radius = 14.875 nm
        offset2 = 14.875 * np.sin(angle)
        for i in range(polymerl):
            polymersurfaceP[k, i, 0] = polymer[i, 0] + offset0
            polymersurfaceP[k, i, 1] = polymer[i, 1] + offset1
            polymersurfaceP[k, i, 2] = polymer[i, 2] + offset2
            polymersurfaceN[k, i, 0] = polymersurfaceP[k, i, 0] + 0.1734 * unit_distance
            polymersurfaceN[k, i, 1] = polymersurfaceP[k, i, 1]
            polymersurfaceN[k, i, 2] = polymersurfaceP[k, i, 2]
    polymersurfaceN = polymersurfaceN.reshape((140,3))
    polymersurfaceP = polymersurfaceP.reshape((140,3))
    # For the protein heads, create one COM per head.
    # Head 0 is at [16.8, 38.0, 20.1]; each subsequent head is shifted 4 nm in x.
    COM0 = np.array([16.8, 38.0, 20.1])
    COMs = np.empty((headl, 3))
    for h in range(headl):
        COMs[h, 0] = COM0[0] + 4.13 * h
        COMs[h, 1] = COM0[1]
        COMs[h, 2] = COM0[2]

    # Create a head surface from a set of offsets (same for every head)
    offsets = np.array([[0.5, 0.0, 0.0],
                        [0.0, 0.5, 0.0],
                        [-0.5, 0.0, 0.0],
                        [-0.0867, -0.492, 0.0],
                        [0.0867, -0.492, 0.0]])
    n_surface = offsets.shape[0]  # here, 5 points per head surface
    headsurfaces = np.empty((headl, n_surface, 3))
    for h in range(headl):
        for i in range(n_surface):
            headsurfaces[h, i, 0] = COMs[h, 0] + offsets[i, 0] * unit_distance
            headsurfaces[h, i, 1] = COMs[h, 1] + offsets[i, 1] * unit_distance
            headsurfaces[h, i, 2] = COMs[h, 2] + offsets[i, 2] * unit_distance

    return polymer, COMs, headsurfaces, polymersurfaceP, polymersurfaceN


#-----------------------------------------------------------
# FORCES
#-----------------------------------------------------------

@njit(fastmath=fastmath, cache=cache)
def morse(a, b, D, maxf):
    sigma = np.log(2)/maxf # maxf is distance at which froce is the greatest
    f_total = np.zeros(3)
    for i in range(a.shape[0]):
        dif0 = a[i, 0] - b[0]
        dif1 = a[i, 1] - b[1]
        dif2 = a[i, 2] - b[2]
        dist = np.sqrt(dif0 * dif0 + dif1 * dif1 + dif2 * dif2)
        if dist == 0:
            continue
        exp_val = np.exp(-sigma * dist)
        f_val = 2 * sigma * D * exp_val * (1.0 - exp_val)   # optimised version of: 2*sigma*D*(np.e**(-sigma*dist))*(1-np.e**(-sigma*dist)), max force = 1/2 * sigma * D, at x = 1/a * ln(2)
        inv_dist = 1.0 / dist
        f_total[0] += f_val * dif0 * inv_dist
        f_total[1] += f_val * dif1 * inv_dist
        f_total[2] += f_val * dif2 * inv_dist
    return f_total


@njit(fastmath=fastmath, cache=cache)
def repulsive(a, b, epsilon=0.1, sigma=1):
    f_total = np.zeros(3)
    for i in range(a.shape[0]):
        dif1 = a[i, 1] - b[1]
        dif2 = a[i, 2] - b[2]
        dist = np.sqrt(dif1 * dif1 + dif2 * dif2)
        if dist>sigma or dist==0:
            continue
        f_val  = -2 * epsilon * (sigma - dist)
        inv_dist = 1.0 / dist
        f_total[1] += f_val * dif1 * inv_dist
        f_total[2] += f_val * dif2 * inv_dist
    return f_total


@njit(fastmath=fastmath, cache=cache)
def repulsive_particle(a, b, epsilon=0.1, sigma=1):
    f_total = np.zeros(3)
    for i in range(a.shape[0]):
        dif0 = a[i, 0] - b[0]
        dif1 = a[i, 1] - b[1]
        dif2 = a[i, 2] - b[2]
        dist = np.sqrt(dif0*dif0 + dif1 * dif1 + dif2 * dif2)
        if dist>sigma or dist==0:
            continue
        f_val  = -2 * epsilon * (sigma - dist)
        inv_dist = 1.0 / dist
        f_total[0] += f_val * dif0 * inv_dist
        f_total[1] += f_val * dif1 * inv_dist
        f_total[2] += f_val * dif2 * inv_dist
    return f_total


@njit(fastmath=fastmath, cache=cache)
def harmonic_angle(surface, COM, n, angles, K=0.01):
    N = surface.shape[0]
    force = np.empty((N, 3))
    for i in range(N):
        j = (i + n) % N
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
def harmonic_surface(surface, COM, epsilon=0.01, sigma=2.065):
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


@njit(fastmath=fastmath, cache=cache)
def neighbour(COM):
    dy = COM[1] - 20
    dz = COM[2] - 20
    theta = math.atan2(dz, dy)
    if theta < 0.0:
        theta += 2.0 * math.pi
        # scale to column index
    inv_bin = 14 / (2.0 * math.pi)
    ida = int(theta * inv_bin)
    if ida >= 14:
        ida = 0
    idx = int(COM[0]/4.13)
    return ida, idx


@njit(fastmath=fastmath, cache=cache)
def get_surface_patch(polysurface, idtheta, idx, wt, wx):
    """
    Return a (2*wt+1)*(2*wx+1) patch of MT surface points around (idtheta, idx),
    spanning wt columns in theta and wx steps along x, with wrap-around.
    """
    num_t = 14
    num_x = 10
    size = (2 * wt + 1) * (2 * wx + 1)
    patch = np.empty((size, 3))
    k = 0
    for dt in range(-wt, wt + 1):
        t = (idtheta + dt) % num_t
        for dx in range(-wx, wx + 1):
            x = (idx + dx) % num_x
            patch[k, :] = polysurface[t, x]
            k += 1
    return patch

#-----------------------------------------------------------
# TIME INTEGRATION FOR MULTIPLE HEADS
#-----------------------------------------------------------
@njit(fastmath=fastmath, cache=cache)
def notverlet_multiple(COMs, surfaces, dt, random_val, mu, polymer, polymersurfaceP, polymersurfaceN, epsi):
    n_heads = COMs.shape[0]
    n_surface = surfaces.shape[1]
    new_COMs = np.empty_like(COMs)
    new_surfaces = np.empty_like(surfaces)
    # For COM forces (from polymer + later head-head interactions)
    COM_forces = np.empty_like(COMs)
    # For surface forces per head
    surface_forces = np.empty_like(surfaces)
    # First, per-head contributions (without head-head interactions)
    for h in range(n_heads):
        force1 = harmonic_angle(surfaces[h], COMs[h], 1, angles)
        force2 = harmonic_angle(surfaces[h], COMs[h], 2, angles)
        force3 = harmonic_angle(surfaces[h], COMs[h], -2, angles)
        force4 = harmonic_angle(surfaces[h], COMs[h], -1, angles)
        angle_forces = force1 + force2 + force3 + force4
        harmonic_bond = harmonic_surface(surfaces[h], COMs[h])
        lj_force_surf = np.zeros_like(surfaces[h])
        #idtheta, idx = neighbour(COMs[h])
        #surfaceneighbourP = get_surface_patch(polymersurfaceP, idtheta, idx, 2, 2)
        #surfaceneighbourN = get_surface_patch(polymersurfaceN, idtheta, idx, 2, 2)
        #lj_force_surf[3] = morse(surfaceneighbourP, surfaces[h, 3], epsi, 1)
        #lj_force_surf[4] = morse(surfaceneighbourN, surfaces[h, 4], epsi, 1)
        lj_force_surf[3] = morse(polymersurfaceP, surfaces[h, 3], epsi, 1)
        lj_force_surf[4] = morse(polymersurfaceN, surfaces[h, 4], epsi, 1)
        for i in range(n_surface):
            surface_forces[h, i, 0] = angle_forces[i, 0] + harmonic_bond[i, 0] + lj_force_surf[i, 0]
            surface_forces[h, i, 1] = angle_forces[i, 1] + harmonic_bond[i, 1] + lj_force_surf[i, 1]
            surface_forces[h, i, 2] = angle_forces[i, 2] + harmonic_bond[i, 2] + lj_force_surf[i, 2]

        # COM force from polymer repulsion
        COM_forces[h] = repulsive(polymer, COMs[h], 0.00005, 16.875)
        harmonic_sum = np.zeros(3)
        for i in range(n_surface):
            harmonic_sum[0] += harmonic_bond[i, 0]
            harmonic_sum[1] += harmonic_bond[i, 1]
            harmonic_sum[2] += harmonic_bond[i, 2]
        COM_forces[h, 0] -= harmonic_sum[0]
        COM_forces[h, 1] -= harmonic_sum[1]
        COM_forces[h, 2] -= harmonic_sum[2]

    # Head-head interactions (compute each pair only once)
    for h in range(n_heads):
        for m in range(h + 1, n_heads):
            a = np.empty((1, 3))
            a[0, 0] = surfaces[m, 2, 0]
            a[0, 1] = surfaces[m, 2, 1]
            a[0, 2] = surfaces[m, 2, 2]
            f = morse(surfaces[m,2].reshape((1,3)), surfaces[h, 0], 0.000001, 1.0)
            surface_forces[h, 0, 0] += f[0]
            surface_forces[h, 0, 1] += f[1]
            surface_forces[h, 0, 2] += f[2]
            surface_forces[m, 2, 0] -= f[0]
            surface_forces[m, 2, 1] -= f[1]
            surface_forces[m, 2, 2] -= f[2]
            a_com = np.empty((1, 3))
            a_com[0, 0] = COMs[m, 0]
            a_com[0, 1] = COMs[m, 1]
            a_com[0, 2] = COMs[m, 2]
            f_com = repulsive_particle(COMs[m].reshape((1,3)), COMs[h], 0.00001, 4.13)
            COM_forces[h, 0] += f_com[0]
            COM_forces[h, 1] += f_com[1]
            COM_forces[h, 2] += f_com[2]
            COM_forces[m, 0] -= f_com[0]
            COM_forces[m, 1] -= f_com[1]
            COM_forces[m, 2] -= f_com[2]
    # Update surfaces and COMs
    for h in range(n_heads):
        for i in range(n_surface):
            for j in range(3):
                new_surfaces[h, i, j] = surfaces[h, i, j] + dt * mu * surface_forces[h, i, j]
        for j in range(3):
            new_COMs[h, j] = COMs[h, j] + dt * (mu * COM_forces[h, j] + random_val[h,j])

        # Box-boundary corrections per head
        if new_COMs[h, 0] < 4.13:
            new_COMs[h, 0] += 12.39
            for i in range(n_surface):
                new_surfaces[h, i, 0] += 12.39
        elif new_COMs[h, 0] > 33.04:
            new_COMs[h, 0] -= 12.39
            for i in range(n_surface):
                new_surfaces[h, i, 0] -= 12.39
        if new_COMs[h, 1] > 100.0:
            new_COMs[h, 1] -= 50.0
            for i in range(n_surface):
                new_surfaces[h, i, 1] -= 50.0
        elif new_COMs[h, 1] < -60.0:
            new_COMs[h, 1] += 50.0
            for i in range(n_surface):
                new_surfaces[h, i, 1] += 50.0
        if new_COMs[h, 2] > 100.0:
            new_COMs[h, 2] -= 50.0
            for i in range(n_surface):
                new_surfaces[h, i, 2] -= 50.0
        elif new_COMs[h, 2] < -60.0:
            new_COMs[h, 2] += 50.0
            for i in range(n_surface):
                new_surfaces[h, i, 2] += 50.0
    return new_COMs, new_surfaces


#-----------------------------------------------------------
# SIMULATION FOR MULTIPLE HEADS IN SEGMENTS
#-----------------------------------------------------------
@njit(fastmath=fastmath, cache=cache)
def simulate_segment(COMs, surfaces, polymer, polymersurfaceP, polymersurfaceN, steps, dt, mu, rand_COM_array, hist_every, epsi):
    histsize = steps // hist_every
    headhist = np.empty((histsize, COMs.shape[0], 3))
    surfacehist = np.empty((histsize, surfaces.shape[0], surfaces.shape[1], 3))
    headhist[0] = COMs
    surfacehist[0] = surfaces
    for i in range(steps):
        COMs, surfaces = notverlet_multiple(COMs, surfaces, dt, rand_COM_array[i], mu, polymer, polymersurfaceP,
                                            polymersurfaceN, epsi)
        if (i % hist_every) == 0:
            headhist[i // hist_every] = COMs
            surfacehist[i // hist_every] = surfaces
        #if i % print_every == 0:
        #    print("Step", i, "/", steps, "COM:", COM[0], COM[1], COM[2])
    return headhist, surfacehist, COMs, surfaces

#----------------------------------------------------------------------------
# Fully njitted segmented simulation driver.
#----------------------------------------------------------------------------
@njit(fastmath=fastmath, cache=cache)
def segmented_simulation(COM, surface, polymer, polymersurfaceP, polymersurfaceN, total_steps, dt, mu, batch_size, random_val, hist_every, epsi, headN):
    num_segments = total_steps // batch_size
    batch_histsize = batch_size // hist_every
    total_histsize = num_segments * batch_histsize
    headhist_full = np.empty((total_histsize,headN, 3))
    surfacehist_full = np.empty((total_histsize, headN, surface.shape[1], 3))
    seg_idx = 0
    current_COM = COM.copy()
    current_surface = surface.copy()
    for seg in range(num_segments):
        rand_COM_array = np.random.uniform(-random_val, random_val, size=(batch_size, headN, 3))
        headhist_batch, surfacehist_batch, final_COM, final_surface = simulate_segment(current_COM, current_surface, polymer, polymersurfaceP, polymersurfaceN, batch_size, dt, mu, rand_COM_array, hist_every, epsi)
        for k in range(batch_histsize):
            headhist_full[seg_idx + k] = headhist_batch[k]
            surfacehist_full[seg_idx + k] = surfacehist_batch[k]
        seg_idx += batch_histsize
        current_COM = final_COM.copy()
        current_surface = final_surface.copy()
    return headhist_full, surfacehist_full


def plotter(headhistv,id, save=True):
    plt.style.use('dark_background')
    render_everyN = 1
    orthogonal_distance = np.linalg.norm(headhistv[::render_everyN, 1:3]-[20,20], axis=1)
    time_arr = np.arange(steps/(render_everyN*hist_every))*render_everyN*hist_every*dt/(steps *10^9)
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(10,10), gridspec_kw={'height_ratios': [3, 1,2]})
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
    ax[2,1].hist(orthogonal_distance, bins=100, range=(15.0,20))
    ax[2,0].hist(orthogonal_distance[1:]-orthogonal_distance[:-1], bins=90)
    gs = ax[1, 0].get_gridspec()
    fig.colorbar(pos, ax=ax[0,:], label='time (µs)', orientation='horizontal', shrink=0.95)
    # remove the underlying Axes
    for ax in ax[1,:]:
        ax.remove()
    axbig = fig.add_subplot(gs[1, :])
    figigi = axbig.plot(time_arr[::], orthogonal_distance[::], linewidth=1)
    axbig.set_ylim(15,50)
    axbig.set_title("orthogonal distance from the center over time", fontsize=8)
    axbig.set_xlabel('time (µs)')
    axbig.set_ylabel('orthogonal distance (nm)')
    plt.show()
    print("shown")
    if save:
        fig.savefig(f'{directory}/figures/engine{id}.png')
    print(f"orthogonal_diff_mean:{(orthogonal_distance[1:]-orthogonal_distance[:-1]).mean()}")


if len(sys.argv) == 1:
    job_id = 0
else:
    job_id = sys.argv[1]

#job_var = np.array()

#-----------------------------------------------------------
# RUN THE SIMULATION (SINGLE CORE)
#-----------------------------------------------------------
polymerLength = 10  # ten repeats
headN = 4 # one head
steps = 1000_000_000
batch_size = 1_000_000
dt = 10  # fs
#random_val = 0.01 # nm/sqrt(fs)
mu = 0.01 #fs mol/g, gives the viscosity of the medium ~ 5.34 mPa*s
print_every = 10_000_000
hist_every = 10_000
unit_distanceX = 4.13 #nm
temp = 310 #K
random_val = np.sqrt(2.0*8.314*(10.0**-3) * mu * temp)/1000.0
epsi = .000001

directory = os.getcwd() #"/data/home/bt23708/Ndc80sim/Ndc80simulator/hpcdump/"

# RUN THE SIM

polymer, COMs, surfaces, polymersurfaceP, polymersurfaceN = polymergenerator(polymerLength, headN)
start = time.perf_counter()
headhist, surfacehist = segmented_simulation(COMs, surfaces, polymer, polymersurfaceP, polymersurfaceN,steps, dt, mu, batch_size, random_val, hist_every, epsi, headN)

np.save(f"{directory}/npy/headhist_{job_id}" , headhist)
print("Simulation complete. Final COM:", headhist[-1], flush=True)
end = time.perf_counter()
print("Elapsed (after compilation) = {}s".format((end - start)), flush=True)

with open(f"{directory}/trajectories/trajectoryn.xyz", "w") as f:
    for t, positionsX in enumerate(np.concatenate(((np.reshape(headhist[::],(int(steps/hist_every),4,1,3))), surfacehist[::]),axis=2)):
        #print(positionsX)
        i = 0
        f.write(f"24 \n \n")
        for positions in positionsX:
            f.write(f"C{1+i} {positions[0,0]:8.4f}{positions[0,1]:8.4f}{positions[0,2]:8.4f} \n")
            f.write(f"C{2+i} {positions[1,0]:8.4f}{positions[1,1]:8.4f}{positions[1,2]:8.4f} \n")
            f.write(f"C{3+i} {positions[2,0]:8.4f}{positions[2,1]:8.4f}{positions[2,2]:8.4f} \n")
            f.write(f"C{4+i} {positions[3,0]:8.4f}{positions[3,1]:8.4f}{positions[3,2]:8.4f} \n")
            f.write(f"C{5+i} {positions[4,0]:8.4f}{positions[4,1]:8.4f}{positions[4,2]:8.4f} \n")
            f.write(f"C{6+i} {positions[5,0]:8.4f}{positions[5,1]:8.4f}{positions[5,2]:8.4f} \n")
            i += 6