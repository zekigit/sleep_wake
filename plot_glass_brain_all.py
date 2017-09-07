from nilearn.plotting import plot_connectome
import numpy as np
import matplotlib.pyplot as plt
from wake_sleep_info import conditions, study_path
import os.path as op
import matplotlib as mpl
from connectivity_fxs import d3_scale

subjects = ['S1', 'S2', 'S3']
markers = ['o', 'D', '^']

cmap = mpl.cm.magma
norm = mpl.colors.Normalize(vmin=0, vmax=0.7)


fig, ax = plt.subplots(2, 1, figsize=(20, 10))
for ix_c, c in enumerate(conditions):
    all_coords = np.loadtxt(op.join(study_path, 'varios', 'all_subj_nodes_%s_gamma-H_for_glass.node' % c))

    for s in range(len(subjects)):
        s_coords = all_coords[all_coords[:, 3] == s + 1, ]
        # s_coords[:, -1] = d3_scale(s_coords[:, -1], out_range=(0., 1))

        plot_connectome(np.zeros([len(s_coords), len(s_coords)]), s_coords[:, :3], node_color=cmap(s_coords[:, -1]), node_size=70,
                        axes=ax[ix_c], node_kwargs={'marker': markers[s], 'edgecolor':None}, black_bg=False, display_mode='lyrz')

cax = fig.add_axes([0.18, 0.495, 0.65, 0.02])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7])
cb.ax.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'])

fig.savefig(op.join(study_path, 'figures', 'glass_brain_all_gamma-H.png'), facecolor='lightgrey', edgecolor='lightgrey', dpi=300)


# plot_connectome(adj, node_coords, node_kwargs={'marker': '*'}, black_bg=True)
