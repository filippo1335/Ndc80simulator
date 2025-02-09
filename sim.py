import sys
import numpy as np

polymerLenght = 10
headN = 1


def polymergenerator(polymerl, headl, mass=1):
    polymer = np.array((range(polymerl), np.zeros(polymerl), np.zeros(polymerl))).T
    # heads = np.array([range(headl), np.zeros(headl), np.zeros(headl)]).T
    heads = np.array((5, 0.15, 0.15))
    head_surface = heads
    head_surface[0] += 1
    centers_of_mass = (head_surface[0] + heads) / 2
    return polymer, heads  # , centers_of_mass, head_surface


def forcefield(a, b, epsilon=0.2, sigma=0.1, mass=1):
    dif = a - b
    dist = np.sqrt(np.sum(np.square(dif), axis=1))
    f = 24 * epsilon * (-2 * (sigma ** 12 / dist ** 13) + (sigma ** 6 / dist ** 7))
    vector = dif / np.array((dist, dist, dist)).T
    forcevector = np.sum(vector * np.array((f, f, f)).T, axis=0)
    a = forcevector
    return forcevector


def verlet(positions, velocities, accelerations, dt):
    # dpositions =
    # positions = positions + np.clip(dpositions, a_min=-1, a_max=1)
    positions = positions + velocities * dt + (accelerations / 2) * (dt ** 2)
    # print(positions.shape)
    # accelerationst1 = np.clip(forcefield(polymerX, positions), a_min=-0.1, a_max=0.1)
    accelerationst1 = forcefield(polymerX, positions) + np.random.uniform(-0.02, 0.02, velocities.shape)
    # dvelocities = (accelerationst1 + accelerations) * (dt/2)
    # velocities =  velocities + np.clip(dvelocities, a_min=-0.1, a_max=0.1)
    velocities = velocities + (accelerationst1 + accelerations) * (
                dt / 2)  # + np.random.uniform(-0.0001, 0.0001, velocities.shape)
    return positions, velocities, accelerationst1

# def ang_integrate(center, surf_1, surf_2, ang_vel, ang_acc, dt, offsets, masses):

polymerX, headX = polymergenerator(polymerLenght, headN)
forcevectors = forcefield(polymerX, headX)
np.set_printoptions(threshold=sys.maxsize)
print(forcevectors)

headX = np.array((5, 0.15, 0.15))

vX = np.array((0,0,0))
aX = np.array((0,0,0))

steps = 200000
printEvery = 1
headhist = np.zeros((int(steps/printEvery), 3))
velhist = np.zeros((int(steps/printEvery), 3))
acchist = np.zeros((int(steps/printEvery), 3))
for i in range(steps):
    headX, vX, aX = verlet(headX, vX, aX, 0.01)
    if i%10000 == 0: print(headX, vX[1], aX[1])
    headhist[i] = headX
    velhist[i] = vX
    acchist[i] = aX

import matplotlib.pyplot as plt
time = np.arange(20000)
# plt.plot(time, headhist[:,0])

#plt.plot(time, acchist[:,0])
#plt.plot(time, velhist[:,0])
plt.figure(figsize=(6, 4))
plt.scatter(headhist[::10,0], headhist[::10,1], cmap="inferno", c=time, linewidth=0.001)
#plt.scatter(velhist[1000:1500:1,0], velhist[1000:1500:1,1], cmap="inferno", c=time, linewidth=0.001)
#plt.scatter(acchist[1000:1500:1,0], acchist[1000:1500:1,1], cmap="inferno", c=time, linewidth=0.001)
print(headhist[::10,0].shape)
print(headhist[::10,1].shape)
print(time.shape)

with open("trajectory.pdb", "w") as f:
    for t, positions in enumerate(headhist[::200]):  # Save every 100th frame
        f.write(f"MODEL     {t + 1}\n")
        f.write(f"ATOM  {1:5d}  C   UNK     1    {positions[0]:8.3f}{positions[1]:8.3f}{positions[2]:8.3f}  1.00  0.00           C\n")
        f.write("ENDMDL\n")

with open("polymerr.pdb", "w") as f:
    for i in range(polymerLenght):
        f.write(f"ATOM  {i+2:5d}  C   UNK     1    {i:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C\n")

headhist[0]