import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import os.path as op
from wake_sleep_info import subjects, conditions, study_path
from connectivity_fxs import binarize
import pandas as pd
import networkx as nx

# Adjust display
pd.set_option('display.expand_frame_repr', False)

subj = 's5'
l = '8s'
ref = 'avg'
fq = 7

conn_files = dict()
results_path = op.join(study_path, subj, 'results', 'pli')
ch_path = op.join(study_path, subj, 'info')

for c in conditions:
    conn_files[c] = (np.load(op.join(results_path, '{}_{}_{}_{}_pli.npz' .format(subj, c, ref, l))))


ch_info = pd.read_pickle('{}/{}/info/{}_{}_info_coords.pkl' .format(study_path, subj, subj, ref))
ch_names = conn_files[c]['ch_names']
ch_used = ch_info[ch_info['Electrode'].isin(ch_names)]

color_codes = pd.factorize(ch_used['Lobe'])[0]
color_labels = pd.factorize(ch_used['Lobe'])[1]

min = np.min(color_codes)
max = np.max(color_codes)

norm = colors.Normalize(vmin=min, vmax=max, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='brg')
color_list = list()
for c in sorted(color_codes):
    color_list.append(mapper.to_rgba(c))

ColorLegend = {'Frontal': 0, 'Parietal': 1, 'Temporal': 2}


fig_comp, axes = plt.subplots(1, 2)

for ix, c in enumerate(conditions):
    con_mat = conn_files[c]['con_mat']
    mat = con_mat[:,:,fq]
    # plt.imshow(mat)
    # plt.colorbar()

    bin_mat = binarize(mat, 85)

    grafo = nx.from_numpy_matrix(bin_mat)

    for nd in range(len(grafo)):
        grafo.node[ix]['spear'] = ch_info['Name'].loc[ix]
        grafo.node[ix]['electrode'] = ch_info['Electrode'].loc[ix]
        grafo.node[ix]['lobe'] = ch_info['Lobe'].loc[ix]
        grafo.node[ix]['matter'] = ch_info['White Grey'].loc[ix]

    nx.draw_networkx(grafo, node_color=color_list, ax=axes[ix], with_labels=False)
    axes[ix].set_title(conditions[ix])

    print(nx.average_clustering(grafo))
    # nx.average_shortest_path_length(grafo)


