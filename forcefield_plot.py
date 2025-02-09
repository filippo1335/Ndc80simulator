import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datashader as ds
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

data = pd.read_table('/Users/filiproch/Downloads/monomer_diffusion/22C/traced/ch1 exp2 22C kymo1-2-1_positions.txt', header=None, usecols= [0, 1, 2])
data.columns = ["time", "x", "brightness"]
data.head()

interpolatedx = interp1d(data['time'], data['x'], kind='slinear')
interpolatedb = interp1d(data['time'], data['brightness'], kind='slinear')


# Generate new time points (including the missing time = 3)
new_time = np.linspace(0, 6, 100)


diff = data.x-data.x.shift(1)

"""
fig, ax1 = plt.subplots()
ax1.scatter(data["x"], data["time"], c=data["brightness"], cmap='gray', marker='s')
ax1.set_facecolor("black")
fig.show()

fig2, ax2 = plt.subplots()
ax2.hist(diff, bins=int(len(diff)/10))
fig2.show()"""


fig3, ax3 = plt.subplots()
ax3.violinplot(diff, showmeans=True)
fig3.show()


def plot_dist(data, bins=100, color='#007E94', sigma=1, scale='linear', title='distribution', xlabel = 'xaxis', ylabel='yaxis', xlim = None, ylim = None):
    # Corrected the typo in 'np.historgam' to 'np.histogram'
    height, bin_edges = np.histogram(data, bins=bins, density=True)

    # The bin_edges define the bin edges, so plotting height vs. the bin centers is usually better
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if sigma != 0:
        smoothed_height = gaussian_filter1d(height, sigma=sigma)
    else:
        smoothed_height = height

    plt.figure(figsize=(5,3))
    plt.plot(bin_centers, smoothed_height, color=color)
    plt.xscale(scale)  # Set the scale before showing the plot
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()

#plot_dist(data.x, bins=50, sigma=10)
