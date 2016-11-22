import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from wake_sleep_info import study_path, conditions, refs, lengths_str
from connectivity_fxs import loadmat
from mpl_toolkits.axes_grid1 import ImageGrid

plt.style.use('ggplot')

# subj = 's2'
# ref = 'bip'
# c = 'wake'
# l = '16s'


def plot_smi_violin(subj):
    pos = [6, 5, 4, 3, 2, 1]
    for ref in refs:
        fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 10), sharex=True, sharey=True)
        for ix_c, c in enumerate(conditions):
            for ix_l, l in enumerate(lengths_str):
                filename = '{}/{}/results/smi/{}_{}_{}_{}_mi.mat'.format(study_path, subj, subj, c, ref, l)
                data = loadmat(filename)
                smi_trials = data['SMI']['Trials']
                smi_avg = data['SMI']['MEAN']
                axes[ix_c, ix_l].violinplot(smi_trials, pos, widths=1, showmeans=True, showextrema=True, showmedians=False,
                                            vert=False, bw_method='silverman')
                axes[ix_c, ix_l].set_xlim([0, 1])
        fig.savefig('{}/{}/figures/{}_{}_violins_smi_trials.eps' .format(study_path, subj, subj, ref), format='eps', dpi=300)
        # plt.boxplot(smi_avg)


def plot_pli_violin(subj):
    for ref in refs:
        fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 10), sharex=True, sharey=True)
        for ix_c, c in enumerate(conditions):
            for ix_l, l in enumerate(lengths_str):
                filename = '{}/{}/results/pli/{}_{}_{}_{}_pli.npz'.format(study_path, subj, subj, c, ref, l)
                data = np.load(filename)

                con = data['con_tril'][:-1, :-1, :]
                con_mat = data['con_mat'][:-1, :-1, :]
                freqs = len(data['freqs'])
                n_ch = len(data['ch_names'])-1

                con = con[np.tril_indices(n_ch, k=-1)]

                values = list()
                con_tril = [x for x in con.T]

                if l == '250ms':
                    pos = [3, 4, 5, 6, 7, 8]
                else:
                    pos = [1, 2, 3, 4, 5, 6, 7, 8]

                # Violin Plot
                axes[ix_c, ix_l].violinplot(con_tril, pos, widths=1, showmeans=True, showextrema=True, showmedians=False,
                                            vert=False, bw_method='silverman')
                axes[ix_c, ix_l].set_xlim([0, 1])
        fig.savefig('{}/{}/figures/{}_{}_violins_pli.eps'.format(study_path, subj, subj, ref), format='eps', dpi=300)  # PLOT


def plot_matrix_conn(subj, conds, ref, l):
    for c in conds:
        filename = '{}/{}/results/pli/{}_{}_{}_{}_pli.npz'.format(study_path, subj, subj, c, ref, l)
        data = np.load(filename)

        con = data['con_tril'][:-1, :-1, :]
        con_mat = data['con_mat'][:-1, :-1, :]
        freqs = len(data['freqs'])
        n_ch = len(data['ch_names']) - 1

        # Matrix Plot
        titles = ['delta', 'theta', 'alpha l ', 'alpha h', 'beta', 'gamma l', 'gamma m', 'gamma h']
        plt.style.use('classic')
        con_fig = plt.figure(figsize=(20, 3))
        grid = ImageGrid(con_fig, 111,
                         nrows_ncols=(1, 8),
                         axes_pad=0.3,
                         cbar_mode='single',
                         cbar_pad='15%',
                         cbar_location='right')

        for idx, ax in enumerate(grid):
            im = ax.imshow(con_mat[:, :, idx], vmin=0, vmax=1)
            ax.set_title(titles[idx])

        cb = con_fig.colorbar(im, cax=grid.cbar_axes[0])
        cb.ax.set_title('wPLI', loc='right')
        # fig.savefig('{}/{}/figures/{}_{}_mats' .format(study_path, subj, subj, ref), format='svg', dpi=300)


if __name__ == '__main__':
    plot_smi('s2')
    plot_pli('s2')

