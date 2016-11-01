from mne.viz import circular_layout, plot_connectivity_circle
from wake_sleep_info import subjects, conditions, results_path, figures_path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os.path as op
import networkx as nx
from connectivity_fxs import binarize, make_bnw_edges
from matplotlib import cm

subj = subjects[2]
cond = conditions[subj][1]
save = True

file_in = op.join(results_path, 'connectivity_{}_{}.npz' .format(subj, cond))
resultados = np.load(file_in)

con = resultados['con_tril']
con_mat = resultados['con_mat']
freqs = len(resultados['freqs'])

# PLOT
titles = ['delta', 'theta', 'alpha', 'beta', 'hi-gamma']

# Matrix Plot
con_fig = plt.figure(figsize=(15, 3))
grid = ImageGrid(con_fig, 111,
                 nrows_ncols=(1, 5),
                 axes_pad=0.3,
                 cbar_mode='single',
                 cbar_pad='10%',
                 cbar_location='right')

for idx, ax in enumerate(grid):
    im = ax.imshow(con_mat[:, :, idx], vmin=0, vmax=1)
    ax.set_title(titles[idx])

cb = con_fig.colorbar(im, cax=grid.cbar_axes[0])
cb.ax.set_title('wPLI', loc='right')

if save:
    con_fig.savefig('{}wPLI_{}_{}.pdf' .format(figures_path, subj, cond), format='pdf', dpi=300)

ch_path = '/Volumes/FAT/Wake_Sleep/images_location/p14_ch_names.csv'
ch_labels = np.genfromtxt(ch_path, delimiter=' ', dtype=None)

# Circular Plot
if subj == 'P14':
    node_names = ch_labels
else:
    node_names = resultados['ch_names']

circ_fig = plt.figure(figsize=(30, 6), facecolor='black')
for idx, fq in enumerate(range(freqs)):
    fq_result = con[:, :, idx]
    im = plot_connectivity_circle(fq_result, node_names=node_names, title=titles[idx], fig=circ_fig, subplot=(1, 5, idx+1),
                                  colorbar=False, colormap='viridis')

if save:
    circ_fig.savefig('{}wPLI_circle_{}_{}.pdf' .format(figures_path, subj, cond), format='pdf', dpi=300)






# Graph Theory
binary_mats_p50 = binarize(con_mat, 50)
binary_mats_p90 = binarize(con_mat, 90)

file_nodes = figures_path + 'bnw/' + 'nodes_{}_{}_' .format(subj, cond)
make_bnw_edges(file_nodes, binary_mats_p90, titles)

graph = nx.from_numpy_matrix(binary_mats_p50[:,:,0])




