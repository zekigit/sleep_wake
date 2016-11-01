from mne.viz import circular_layout, plot_connectivity_circle
from ita_info import conditions, results_path, figures_path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import mne
import numpy as np
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import axes3d
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.io.matlab import savemat
import numpy as np
import os.path as op

# import networkx as nx
# from connectivity_fxs import binarize, make_bnw_edges
# from matplotlib import cm

cond = 'wake'
save = True

file_in = op.join(results_path, 'connectivity_EGI_{}.npz' .format(cond))
resultados = np.load(file_in)

con = resultados['con_tril']
con_mat = resultados['con_mat']
freqs = len(resultados['freqs'])

# PLOT
titles = ['delta (1-4hz)', 'theta (4-8hz)', 'alpha I (8-10hz)', 'alpha II (10-13hz)', 'beta(13-30hz)']

savemat('mila_wpli_wake.mat', mdict={'con_mat': con_mat})

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

# node_names = resultados['ch_names']
#
# circ_fig = plt.figure(figsize=(30, 6), facecolor='black')
# for idx, fq in enumerate(range(freqs)):
#     fq_result = con[:, :, idx]
#     im = plot_connectivity_circle(fq_result, node_names=node_names, title=titles[idx], fig=circ_fig, subplot=(1, 5, idx+1),
#                                   colorbar=False, colormap='viridis')
#
#
# if save:
#     circ_fig.savefig('{}wPLI_circle_{}.pdf' .format(figures_path, cond), format='pdf', dpi=300)
# plt.show()

fq=0
res = con_mat[:,:,fq]
mean_x_ch = np.mean(res, axis=1)

montage = mne.channels.read_montage('GSN-HydroCel-256')
# montage.plot(show_names=True)

x = montage.pos[:-9,0]
y = montage.pos[:-9,1]
z = montage.pos[:-9,2]

Z = np.outer(z.T, z)
X, Y = np.meshgrid(x,y)

color_dim = X

minn, maxx = color_dim.min(), color_dim.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = m.to_rgba(color_dim)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z, rstride=1, cstride=1, vmin=minn, vmax=maxx, shade=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.canvas.show()

# X, Y = np.meshgrid(X,Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.scatter3D(X, Y, Z, c=mean_x_ch, s=60)
ax.plot_surface(x,y,z, rstride=1, cstride=1)

