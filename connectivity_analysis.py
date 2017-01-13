import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sps
from wake_sleep_info import study_path, conditions, refs, lengths_str
from connectivity_fxs import loadmat, reshape_wsmi, make_bnw_nodes
from mpl_toolkits.axes_grid1 import ImageGrid
from ieeg_fx import load_pli, calc_electrode_dist


# Adjust display
pd.set_option('display.expand_frame_repr', False)


def plot_smi_violin_all(subj):
    plt.style.use('ggplot')
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


def plot_pli_violin_all(subj):
    plt.style.use('ggplot')
    for ref in refs:
        fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 10), sharex=True, sharey=True)
        for ix_c, c in enumerate(conditions):
            for ix_l, l in enumerate(lengths_str):
                con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli(study_path, subj, c, ref, l)
                # Violin Plot
                axes[ix_c, ix_l].violinplot(con_tril, pos, widths=1, showmeans=True, showextrema=True, showmedians=False,
                                            vert=False, bw_method='silverman')
                axes[ix_c, ix_l].set_xlim([0, 1])
        fig.savefig('{}/{}/figures/{}_{}_violins_pli.eps'.format(study_path, subj, subj, ref), format='eps', dpi=300)  # PLOT


def plot_pli_violin(subj, ref, l):
    plt.style.use('ggplot')
    print('pli violin')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), sharex=True, sharey=True)
    for ix_c, c in enumerate(conditions):
            con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli(study_path, subj, c, ref, l)

            # Violin Plot
            axes[ix_c].violinplot(con_tril, pos, widths=1, showmeans=True, showextrema=True, showmedians=False,
                                  vert=False, bw_method='silverman')
            axes[ix_c].set_xlim([0, 1])
    fig.savefig('{}/{}/figures/{}_{}_violin_ind_pli_{}.eps'.format(study_path, subj, subj, ref, l), format='eps', dpi=300)  # PLOT


def plot_smi_violin(subj, ref, l):
    plt.style.use('ggplot')
    print('smi violin')
    pos = [6, 5, 4, 3, 2, 1]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), sharex=True, sharey=True)
    for ix_c, c in enumerate(conditions):
        filename = '{}/{}/results/smi/{}_{}_{}_{}_mi.mat'.format(study_path, subj, subj, c, ref, l)
        data = loadmat(filename)
        smi_trials = data['SMI']['Trials']
        smi_avg = data['SMI']['MEAN']
        axes[ix_c].violinplot(smi_trials, pos, widths=1, showmeans=True, showextrema=True, showmedians=False,
                              vert=False, bw_method='silverman')
        axes[ix_c].set_xlim([0, 1])
    fig.savefig('{}/{}/figures/{}_{}_violins_ind_smi_trials_{}.eps' .format(study_path, subj, subj, ref, l), format='eps', dpi=300)
    # plt.boxplot(smi_avg)


def plot_matrix_pli(subj, conds, ref, l):
    plt.style.use('classic')
    print('pli matrix')
    for c in conds:
        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli(study_path, subj, c, ref, l)

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
        con_fig.savefig('{}/{}/figures/{}_{}_{}_pli_{}_conmat.eps' .format(study_path, subj, subj, c, ref, l), format='eps', dpi=300)


def plot_matrix_smi(subj, conds, ref, l):
    plt.style.use('classic')
    print('smi matrix')
    titles = ['2ms', '4ms', '8ms', '16ms', '32ms', '64ms']

    for c in conds:
        filename = '{}/{}/results/smi/{}_{}_{}_{}_mi.mat'.format(study_path, subj, subj, c, ref, l)
        data = loadmat(filename)
        smi_trials = data['SMI']['Trials']
        smi_avg = data['SMI']['MEAN']
        n_ch = reshape_wsmi(smi_avg[0])[1]

        con_mat = np.empty((n_ch, n_ch, len(smi_avg)))
        for tau in range(len(smi_avg)):
            con_mat[:, :, tau] = reshape_wsmi(smi_avg[tau])[0]

        con_fig = plt.figure(figsize=(20, 3))
        grid = ImageGrid(con_fig, 111,
                         nrows_ncols=(1, 6),
                         axes_pad=0.3,
                         cbar_mode='single',
                         cbar_pad='15%',
                         cbar_location='right')

        for idx, ax in enumerate(grid):
            im = ax.imshow(con_mat[:, :, idx], vmin=0, vmax=1)
            ax.set_title(titles[idx])

        cb = con_fig.colorbar(im, cax=grid.cbar_axes[0])
        cb.ax.set_title('SMI', loc='right')
        con_fig.savefig('{}/{}/figures/{}_{}_{}_smi_{}_conmat.eps'.format(study_path, subj, subj, c, ref, l), format='eps', dpi=300)


def pli_by_distance(subj, ref, win):
    ch_info = pd.read_pickle('{}/{}/info/{}_{}_info_coords.pkl' .format(study_path, subj, subj, ref))
    pairs, distances, dist_mat = calc_electrode_dist(ch_info)
    dists = np.array(distances)
    dists /= 10  # cm
    for c in conditions:
        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli(study_path, subj, c, ref, win)

        titles = ['delta', 'theta', 'alpha l ', 'alpha h', 'beta', 'gamma l', 'gamma m', 'gamma h']
        plt.style.use('classic')
        con_fig, axes = plt.subplots(1, 8, sharex=True, sharey=True, figsize=(20, 8))

        for idx, ax in enumerate(axes):
            ax.scatter(dists, con_tril[idx])
            ax.set_title(titles[idx])
            ax.set_ylim([0, 1])

        # cb = con_fig.colorbar(im, cax=grid.cbar_axes[0])
        # cb.ax.set_title('wPLI', loc='right')
        con_fig.savefig('{}/{}/figures/{}_{}_{}_pli_dist_{}_conmat.eps' .format(study_path, subj, subj, c, ref, win), format='eps', dpi=300)


def plot_da(subj):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    for ix, c in enumerate(conditions):
        da_raw = loadmat('{}/{}/results/{}_{}_bip_DA.mat'.format(study_path, subj, subj, c))
        da = da_raw['DA']
        mean_da = np.mean(da, 0)
        filt_da = sps.savgol_filter(mean_da, 15, 3)
        axes[ix].plot(mean_da, 'b')
        axes[ix].plot(filt_da, 'g', linewidth=3)
        axes[ix].set_title(c)
        axes[ix].set_ylim([2, 8])


def create_node_file(subj, ref):
    ch_info = pd.read_pickle('{}/{}/info/{}_{}_info_coords.pkl' .format(study_path, subj, subj, ref))
    coords = ch_info[['natX', 'natY', 'natZ']].values
    node_file = '{}/{}/info/{}_{}_nodes.node' .format(study_path, subj, subj, ref)
    make_bnw_nodes(file_nodes=node_file, channels=coords, colors=1.0, sizes=0.5)


if __name__ == '__main__':
    for s in ['s5']:
        for r in ['avg']:
            print('Subject: {}' .format(s))

            # plot_pli_violin_all(s)
            # plot_pli_violin(s, r, '8s')
            # plot_matrix_pli(s, conditions, r, '8s')
            pli_by_distance(s, r, '8s')
            # plot_smi_violin_all(s)
            # plot_smi_violin(s, r, '500ms')
            # plot_matrix_smi(s, conditions, r, '500ms')
