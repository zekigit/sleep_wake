import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sps
import scipy.io as scio
import mne
from scipy import stats
from wake_sleep_info import study_path, conditions, refs, lengths_str, frequencies, n_jobs
from connectivity_fxs import loadmat, reshape_wsmi, make_bnw_nodes, find_threshold, binarize_mat, calc_graph_metrics, graph_threshold, \
    d3_scale
from mpl_toolkits.axes_grid1 import ImageGrid
from ieeg_fx import load_pli_ieeg, calc_electrode_dist, load_pli, select_gm_chans
from mne.time_frequency import psd_multitaper
from study_fx import permutation_t_test
from statsmodels.sandbox.stats.multicomp import multipletests
from collections import OrderedDict

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
                con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, l)
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
            con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, l)

            # Violin Plot
            axes[ix_c].violinplot(con_tril, pos, widths=1, showmeans=True, showextrema=True, showmedians=False,
                                  vert=False, bw_method='silverman')
            axes[ix_c].set_xlim([0, 1])
    fig.savefig('{}/{}/figures/{}_{}_violin_ind_pli_{}.eps'.format(study_path, subj, subj, ref, l), format='eps', dpi=300)  # PLOT


def plot_pli_violin_gamma(subj, ref, l):
    plt.style.use('ggplot')
    print('pli violin')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), sharex=True, sharey=True)
    for ix_c, c in enumerate(conditions):
            con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, l)
            gm_chans, gm_names, gm_ch_info = select_gm_chans(subj, ref, study_path, ch_names)
            con_gm = con_mat[gm_chans[:, None], gm_chans, :]
            con_gm_gamma = con_gm[:, :, [5, 6, 7]]
            pos = pos[0:3]
            tril_gm_gamma = con_gm_gamma[np.tril_indices(con_gm.shape[0], k=-1)]

            # Violin Plot
            axes[ix_c].violinplot(tril_gm_gamma, pos, widths=1, showmeans=False, showextrema=True, showmedians=True,
                                  vert=False, bw_method='silverman')
            axes[ix_c].set_xlim([0, 1])
    fig.savefig('{}/{}/figures/{}_{}_violin_gamma_pli_{}.eps'.format(study_path, subj, subj, ref, l), format='eps', dpi=300)  # PLOT


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
        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, l)

        gm_chans, gm_names, gm_ch_info = select_gm_chans(subj, ref, study_path, ch_names)

        # Matrix Plot
        titles = ['delta', 'theta', 'alpha l ', 'alpha h', 'beta', 'gamma l', 'gamma m', 'gamma h']
        con_fig = plt.figure(figsize=(20, 3))
        grid = ImageGrid(con_fig, 111,
                         nrows_ncols=(1, 8),
                         axes_pad=0.3,
                         cbar_mode='single',
                         cbar_pad='15%',
                         cbar_location='right')

        for idx, ax in enumerate(grid):
            im = ax.imshow(con_mat[gm_chans[:, None], gm_chans, idx], vmin=0, vmax=1, cmap='jet')
            ax.set_title(titles[idx])

        cb = con_fig.colorbar(im, cax=grid.cbar_axes[0])
        cb.ax.set_title('wPLI', loc='right')
        con_fig.savefig('{}/{}/figures/{}_{}_{}_pli_{}_conmat.eps' .format(study_path, subj, subj, c, ref, l), format='eps', dpi=300)


def plot_matrix_pli_anotomical(subj, conds, ref, l):
    plt.style.use('classic')
    print('pli matrix anatomical')
    lobes_labels = ['Frontal', 'Temporal', 'Parietal', 'Occipital']

    for c in conds:
        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, l)

        gm_chans, gm_names, gm_ch_info = select_gm_chans(subj, ref, study_path, ch_names)

        # sort by anatomy
        gm_ch_info['Original_ix'] = np.arange(0, len(gm_ch_info))

        lobes = [x[1] for x in gm_ch_info['Lobe'].str.split().tolist()]
        gm_ch_info['LobePlot'] = pd.Categorical(lobes, lobes_labels)
        gm_ch_info_ordered = gm_ch_info.sort_values(by=['LobePlot', 'Electrode'])

        reorder_ix = np.array(gm_ch_info_ordered.Original_ix.tolist())
        lobe_division = [ix for ix, x in enumerate(gm_ch_info_ordered.LobePlot.tolist()) if
                         (ix < len(gm_ch_info_ordered.LobePlot)-2) and (
                         gm_ch_info_ordered.LobePlot.iloc[ix] != gm_ch_info_ordered.LobePlot.iloc[ix+1])]

        # Matrix Plot
        titles = ['delta', 'theta', 'alpha l ', 'alpha h', 'beta', 'gamma l', 'gamma m', 'gamma h']
        con_fig = plt.figure(figsize=(20, 3))
        grid = ImageGrid(con_fig, 111,
                         nrows_ncols=(1, 8),
                         axes_pad=0.3,
                         cbar_mode='single',
                         cbar_pad='15%',
                         cbar_location='right')

        for idx, ax in enumerate(grid):
            con_gm = con_mat[gm_chans[:, None], gm_chans, idx]
            con_gm_ordered = con_gm[reorder_ix][:, reorder_ix]

            im = ax.imshow(con_gm_ordered, vmin=0, vmax=1, cmap='magma')
            ax.vlines(lobe_division, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1])
            ax.hlines(lobe_division, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1])
            ax.set_title(titles[idx])

        cb = con_fig.colorbar(im, cax=grid.cbar_axes[0])
        cb.ax.set_title('wPLI', loc='right')

        print(gm_ch_info_ordered.LobePlot.unique())
        con_fig.savefig('{}/{}/figures/{}_{}_{}_pli_{}_conmat_anatomical.eps' .format(study_path, subj, subj, c, ref, l), format='eps', dpi=300)

    node_file = op.join(study_path, subj, 'info', '%s_nodes_by_lobe.node' % subj)
    coords = gm_ch_info_ordered[['natX', 'natY', 'natZ']].values
    colors = [1 if x == 'Frontal' else 2 if x == 'Temporal' else 3 if x == 'Parietal' else 4 for x in gm_ch_info_ordered.LobePlot.tolist()]
    make_bnw_nodes(node_file, coords, colors, [3]*len(coords))


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
    plt.imshow(dist_mat)
    plt.colorbar()
    for c in conditions:
        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, win)

        titles = ['delta', 'theta', 'alpha l ', 'alpha h', 'beta', 'gamma l', 'gamma m', 'gamma h']
        plt.style.use('classic')
        con_fig, axes = plt.subplots(1, 8, sharex=True, sharey=True, figsize=(20, 8))

        for idx, ax in enumerate(axes):
            ax.scatter(dists, con_tril[idx])
            ax.set_title(titles[idx])
            ax.set_ylim([0, 1])

        con_fig.savefig('{}/{}/figures/{}_{}_{}_pli_dist_{}_conmat.eps' .format(study_path, subj, subj, c, ref, win), format='eps', dpi=300)


def pli_by_distance_gamma(subj, ref, win):
    dfs = list()
    for c in conditions:
        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, win)
        gm_chans, gm_names, gm_ch_info = select_gm_chans(subj, ref, study_path, ch_names)
        con_gm = con_mat[gm_chans[:, None], gm_chans, :]
        con_gm_gamma = con_gm[:, :, [5, 6, 7]]
        tril_gm_gamma = con_gm_gamma[np.tril_indices(con_gm.shape[0], k=-1)]

        pairs, distances, dist_mat = calc_electrode_dist(gm_ch_info)
        dists = np.array(distances)
        dists /= 10  # cm
        # plt.imshow(dist_mat)
        # plt.colorbar()

        titles = ['gamma-L', 'gamma-M', 'gamma-H']
        plt.style.use('classic')
        con_fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 8))

        for idx, ax in enumerate(axes):
            ax.scatter(dists, tril_gm_gamma[:, idx])
            ax.set_title(titles[idx])
            ax.set_ylim([0, 1])

        dfs.append(pd.DataFrame({'Pair': pairs, 'Distance': dists, 'low-gamma': tril_gm_gamma[:, 0], 'mid-gamma': tril_gm_gamma[:, 1],
                                'high-gamma': tril_gm_gamma[:, 2], 'Condition': c}))

        con_fig.savefig('{}/{}/figures/{}_{}_{}_pli_gamma_dist_{}_conmat.eps' .format(study_path, subj, subj, c, ref, win), format='eps', dpi=300)

    dist_df = pd.concat(dfs)
    dist_name = '{}/{}/results/tables/{}_gamma_dist_table.csv'.format(study_path, subj, subj)
    dist_df.to_csv(dist_name)


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
        fig.savefig('{}/{}/figures/{}_DA.eps' .format(study_path, subj, subj), format='eps', dpi=300)


def create_node_file(subj, ref):
    ch_info = pd.read_pickle('{}/{}/info/{}_{}_info_coords.pkl' .format(study_path, subj, subj, ref))
    coords = ch_info[['natX', 'natY', 'natZ']].values
    node_file = '{}/{}/info/{}_{}_nodes.node' .format(study_path, subj, subj, ref)
    make_bnw_nodes(file_nodes=node_file, coords=coords, colors=1.0, sizes=0.5)


def create_edge_file(subj, ref, cond):
    con_mat = np.load('{}/{}/results/pli/{}_{}_{}_8s_pli.npz' .format(study_path, subj, subj, cond, ref))['con_mat']
    titles = ['delta', 'theta', 'alpha l ', 'alpha h', 'beta', 'gamma l']
    for ix, fq in enumerate(titles):
        file_nodes = '{}/{}/results/pli/{}_8s_conmat_{}.edge' .format(study_path, subj, subj, fq)
        np.savetxt(file_nodes, con_mat[:, :, ix], delimiter='\t')


def plot_pli_matrix_scalp(subj):
    plt.style.use('classic')
    print('pli matrix')
    for c in conditions:
        filename = op.join(study_path, subj, 'results', 'pli', '{}_{}_8s_pli_scalp_CSD.npz' .format(subj, c))
        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli(filename)

        # Matrix Plot
        titles = ['delta', 'theta', 'alpha l ', 'alpha h', 'beta', 'gamma l']
        plt.style.use('classic')
        con_fig = plt.figure(figsize=(20, 3))
        grid = ImageGrid(con_fig, 111,
                         nrows_ncols=(1, 6),
                         axes_pad=0.3,
                         cbar_mode='single',
                         cbar_pad='15%',
                         cbar_location='right')

        for idx, ax in enumerate(grid):
            im = ax.imshow(con_mat[:, :, idx], vmin=0, vmax=1, cmap='viridis')
            ax.set_title(titles[idx])

        cb = con_fig.colorbar(im, cax=grid.cbar_axes[0])
        cb.ax.set_title('wPLI', loc='right')
        con_fig.savefig('{}/{}/figures/{}_{}_8s_pli_scalp_conmat_CSD.eps' .format(study_path, subj, subj, c), format='eps', dpi=300)

        exp_data = dict(con=con,
                        con_mat=con_mat,
                        con_tril=con_tril,
                        freqs=freqs,
                        ch_names=ch_names)
        file_mlab = filename.replace("npz", "mat")
        print('saving ' + file_mlab)
        scio.savemat(file_mlab, exp_data)


def create_graph_nodes(subj, ref, cond):
    l = '8s'
    con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, cond, ref, l)
    band = 2
    mat = con_mat[:, :, band].copy()
    threshold = find_threshold(mat)
    bin_mat = binarize_mat(mat, threshold)

    clustering, degree, graph = calc_graph_metrics(bin_mat, ch_names)
    # nx.draw_networkx(graph)

    # load chans, select and order by appearance on ch_names of pli result
    ch_info = pd.read_pickle('{}/{}/info/{}_{}_info_coords.pkl' .format(study_path, subj, subj, ref))
    ch_info = ch_info[ch_info['Electrode'].isin(ch_names)]
    sorter_index = dict(zip(ch_names, range(len(ch_names))))
    ch_info['sorter'] = ch_info['Electrode'].map(sorter_index)
    ch_info.sort_values(['sorter'], ascending=[True], inplace=True)

    coords = ch_info[['natX', 'natY', 'natZ']].values
    node_file = '{}/{}/info/{}_{}_{}_nodes_degree.node' .format(study_path, subj, subj, ref, cond)
    sizes = list(degree.values())
    colors = list(degree.values())
    make_bnw_nodes(node_file, coords, colors, sizes)

    # sizes_cl = list(clustering)


def export_deg_table(subj, ref):
    l = '8s'
    results = list()
    values = list()
    for c in conditions:
        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, l)
        results.append(con_mat)
        values.append(con)

    gm_chans, gm_names, gm_ch_info = select_gm_chans(subj, ref, study_path, ch_names)
    #
    # # load chans, select and order by appearance on ch_names of pli result
    # ch_info = pd.read_pickle('{}/{}/info/{}_{}_info_coords.pkl' .format(study_path, subj, subj, ref))
    # ch_info = ch_info[ch_info['Electrode'].isin(ch_names)]
    # sorter_index = dict(zip(ch_names, range(len(ch_names))))
    # ch_info['sorter'] = ch_info['Electrode'].map(sorter_index)
    # ch_info.sort_values(['sorter'], ascending=[True], inplace=True)
    #
    # grey_matter_chans = ch_info[ch_info['White Grey'] == 'Grey Matter']
    # gm_chans = np.array([ix for ix, ch in enumerate(ch_names) if ch in grey_matter_chans['Electrode'].values])
    # n_gm_chans = len(gm_chans)
    # gm_names = grey_matter_chans['Electrode'].values

    # coords = grey_matter_chans[['natX', 'natY', 'natZ']].values
    coords = gm_ch_info[['natX', 'natY', 'natZ']].values
    coords_add = np.vstack((coords, np.array([[70, 70, 70], [70, 70, 70]])))

    thresholds = list()
    for ix, f in enumerate(frequencies):
        fq_vals = np.concatenate((values[0][gm_chans, ix], values[1][gm_chans, ix]))
        median = np.median(fq_vals)
        sd = np.std(fq_vals)
        threshold = median
        thresholds.append(threshold)

    # bin_mats = np.empty((n_gm_chans, n_gm_chans, freqs, len(conditions)))
    bin_mats = np.empty((len(gm_chans), len(gm_chans), freqs, len(conditions)))
    conds = list()
    fqs = list()
    chans = list()
    degrees = list()
    steps = np.linspace(0, 1, 100)
    avgs = np.empty((len(frequencies), len(conditions), len(steps)))
    sds = np.empty((len(frequencies), len(conditions), len(steps)))

    dfs = list()

    for ix_c, c in enumerate(conditions):
        for ix_f, f in enumerate(frequencies):
            mat = results[ix_c][:, :, ix_f]
            gm_mat = mat[gm_chans[:, None], gm_chans]
            bin_mat = binarize_mat(gm_mat, thresholds[ix_f])
            clustering, degree, graph = calc_graph_metrics(bin_mat, gm_names)
            bin_mats[:, :, ix_f, ix_c] = bin_mat.copy()
            degree_norm = np.array(list(degree.values()))/len(degree) * 100
            degrees.extend(list(degree_norm))
            chans.extend(gm_names)
            fqs.extend([f] * len(degree))
            conds.extend([c] * len(degree))

            avg, sd, deg_df = graph_threshold(gm_mat, steps)
            avgs[ix_f, ix_c, :] = avg.copy()
            sds[ix_f, ix_c, :] = sd.copy()
            deg_df['Condition'] = c
            deg_df['Frequency'] = f
            deg_df['Threshold'] = np.repeat(np.arange(1,101), len(gm_chans), axis=0)
            dfs.append(deg_df)

            node_file = '{}/{}/results/node_files/{}_{}_{}_{}_nodes_degree.node'.format(study_path, subj, subj, ref, c, f)
            sizes = 3.0
            degree_norm = np.append(degree_norm, np.array([[0], [1]]))
            colors = list(degree_norm)
            make_bnw_nodes(node_file, coords_add, colors, sizes)

            node_file = '{}/{}/results/node_files/{}_{}_{}_{}_nodes_avgPLI.node'.format(study_path, subj, subj, ref, c, f)
            ch_avg = np.mean(gm_mat, axis=1)
            ch_avg = np.append(ch_avg, np.array([[0], [0.7]]))
            colors = list(ch_avg)
            make_bnw_nodes(node_file, coords_add, colors, sizes)

    deg_dist = pd.concat(dfs)
    # gammas = ['gamma-L', 'gamma-M', 'gamma-H']
    fq_ixs = [0, 1, 2, 3, 4, 5, 6, 7]

    n_perm = 1000
    ts_list = list()
    thr_list = list()
    for f in frequencies:
        ext_perms = np.empty((len(steps), 2))
        t_perms = list()
        ts = list()
        for st in range(len(steps)):
            wake = deg_dist['Degree'].loc[(deg_dist['Condition'] == 'wake') & (deg_dist['Frequency'] == f) & (deg_dist['Threshold'] == st+1)].values
            sleep = deg_dist['Degree'].loc[(deg_dist['Condition'] == 'sleep') & (deg_dist['Frequency'] == f) & (deg_dist['Threshold'] == st+1)].values
            t, p = stats.ttest_ind(wake, sleep, equal_var=False)
            ts.append(t)
            t_list = list()
            for per in range(n_perm):
                joint = np.concatenate((wake, sleep))
                np.random.shuffle(joint)
                split = np.split(joint, 2)
                t_perm, p_perm = stats.ttest_ind(split[0], split[1], equal_var=False)
                t_list.append(t_perm)
            t_perms.append(t_list)
            ext_perms[st, 0] = np.amin(t_list)
            ext_perms[st, 1] = np.amax(t_list)
        low_t = np.percentile(ext_perms[~np.isnan(ext_perms)], 2.5)
        high_t = np.percentile(ext_perms[~np.isnan(ext_perms)], 97.5)
        thr_list.append((low_t, high_t))
        ts_list.append(ts)

    degree_df = pd.DataFrame({'Condition': conds, 'Frequency': fqs, 'Channel': chans, 'Degree': degrees, 'Subject': subj})
    save_path = op.join(study_path, subj, 'results', 'tables', '{}_degree_table.csv' .format(subj))
    degree_df.to_csv(save_path)

    # Plot by threshold
    plt.style.use('ggplot')
    thr_fig, axes = plt.subplots(2, 8, sharex=True, sharey=False, gridspec_kw={'height_ratios': [5, 1]})

    for ix_p, (f, ix_f) in enumerate(zip(frequencies, fq_ixs)):
        for cond, color in zip([0, 1], ['g', 'b']):
            axes[0][ix_p].plot(steps, avgs[ix_f, cond, :], color)
            axes[0][ix_p].fill_between(steps, avgs[ix_f, cond, :] - sds[ix_f, cond, :],
                                       avgs[ix_f, cond, :] + sds[ix_f, cond, :], facecolor=color, alpha=0.5)

        # axes[0][ix_p].axvline(x=thresholds[ix_f], color='k', linestyle='--')
        axes[0][ix_p].set_title(f)
        axes[0][ix_p].set_ylim([-0.1, 1])
        axes[1][ix_p].plot(steps, ts_list[ix_p], color='grey')
        axes[1][ix_p].fill_between(steps, 0, ts_list[ix_p], where=ts_list[ix_p] > thr_list[ix_p][1], facecolor='g', alpha=0.5)
        axes[1][ix_p].fill_between(steps, 0, ts_list[ix_p], where=ts_list[ix_p] < thr_list[ix_p][0], facecolor='b', alpha=0.5)
        # axes[1][ix_p].axvline(x=thresholds[ix_f], color='k', linestyle='--')
        axes[1][ix_p].set_ylim([-15, 40])
    thr_fig.savefig('{}/{}/figures/{}_{}_thr_deg_all.eps'.format(study_path, subj, subj, ref), format='eps', dpi=300)  # PLOT

    for ix_f, f in enumerate(frequencies):
        c1_deg = np.array(degree_df.loc[(degree_df['Condition'] == 'wake') & (degree_df['Frequency'] == f), 'Degree'])
        c2_deg = np.array(degree_df.loc[(degree_df['Condition'] == 'sleep') & (degree_df['Frequency'] == f), 'Degree'])
        diff_deg = (c1_deg-c2_deg)*100
        node_file_diff = '{}/{}/results/graph_theory/{}_{}_{}_nodes_degree_diff.node'.format(study_path, subj, subj, ref, f)
        sizes = 3.0
        colors = list(diff_deg)
        make_bnw_nodes(node_file_diff, coords, colors, sizes)


def conn_hfo_analysis(subj):
    ref = 'avg'
    win = '8s'
    lobes_labels = ['Frontal', 'Temporal', 'Parietal', 'Occipital']
    rec_durations = {'s1': {'wake': 23.93, 'sleep': 20.0}, 's2': {'wake': 13.33, 'sleep': 14.93}, 's3': {'wake': 24.80, 'sleep': 24.80}}

    df_x_conds = list()
    for c in conditions:
        hfos = pd.read_csv('{}/{}/results/tables/{}_{}_hfo_table.csv'.format(study_path, subj, subj, c),
                           names=['channel', 'hfo_gamma', 'hfo_ripples', 'hfo_fast_ripples', 'hfo_spikes', 'hfo_other'])

        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, win)
        gm_chans, gm_names, gm_ch_info = select_gm_chans(subj, ref, study_path, ch_names)


        # sort by anatomy
        gm_ch_info['Original_ix'] = np.arange(0, len(gm_ch_info))

        lobes = [x[1] for x in gm_ch_info['Lobe'].str.split().tolist()]
        gm_ch_info['LobePlot'] = pd.Categorical(lobes, lobes_labels)
        gm_ch_info_ordered = gm_ch_info.sort_values(by=['LobePlot', 'Electrode'])

        reorder_ix = np.array(gm_ch_info_ordered.Original_ix.tolist())

        channels = [ch.replace('\'', '_i') for ch in gm_ch_info_ordered.Electrode.tolist()]
        # channels = [ch.replace('\'', '_i') for ch in gm_names]
        con_gm = con_mat[gm_chans[:, None], gm_chans, :]

        con_gm_ordered = con_gm[reorder_ix, :, :][:, reorder_ix, :]

        if subj == 's2':
            if c == 'wake':
                remove = 'A10'
            elif c == 'sleep':
                remove = 'A6'
            ix_remove = channels.index(remove)
            channels.remove(remove)
            con_gm_ordered = np.delete(np.delete(con_gm_ordered, ix_remove, axis=0), ix_remove, axis=1)
            gm_ch_info_ordered = gm_ch_info_ordered[gm_ch_info_ordered.Electrode != remove]

        # Channel Avergares
        ch_avgs = np.mean(con_gm_ordered, 1)
        conn_dat = pd.DataFrame(OrderedDict((('channel', channels), ('delta', ch_avgs[:, 0]), ('theta', ch_avgs[:, 1]),
                                             ('alpha-L', ch_avgs[:, 2]), ('alpha-H', ch_avgs[:, 3]), ('beta', ch_avgs[:, 4]),
                                             ('gamma-L', ch_avgs[:, 5]), ('gamma-M', ch_avgs[:, 6]), ('gamma-H', ch_avgs[:, 7]),
                                             ('x', gm_ch_info_ordered.natX), ('y', gm_ch_info_ordered.natY),
                                             ('z', gm_ch_info_ordered.natZ), ('area', gm_ch_info_ordered['Area Specific']))))

        df = pd.merge(conn_dat, hfos, on='channel')

        # if (subj == 's2') and (c == 'wake'):
        #     df = df[df.channel != 'A10']
        # elif (subj == 's2') and (c == 'sleep'):
        #     df = df[df.channel != 'A6']

        df.to_csv('{}/{}/results/tables/{}_{}_regr_table.csv' .format(study_path, subj, subj, c))

        # Export for stats and plot by areas
        con_values = con_gm_ordered.copy()
        [np.fill_diagonal(con_values[..., ix], np.nan) for ix in range(con_values.shape[2])]
        con_values = np.reshape(con_values[np.where(~np.eye(con_values.shape[0], dtype=bool))],
                                (con_values.shape[0], con_values.shape[0]-1, con_values.shape[2]))

        df_cond = list()
        for ix_f, f in enumerate(frequencies):
            df_x_roi = pd.concat([pd.DataFrame({'channel': channels, 'cond': c, 'freq': f}), pd.DataFrame(con_values[:, :, ix_f])], axis=1)
            df_cond.append(df_x_roi)

        df_x_conds.append(pd.concat(df_cond))
    df_area_export = pd.concat(df_x_conds, ignore_index=True)
    df_area_export.to_csv(op.join(study_path, subj, 'results', 'tables', '%s_df_x_chan.csv' % subj))


def power_analysis(subj):
    cond_data = list()
    n_perm = 10e3
    for c in conditions:
        file = op.join(study_path, subj, 'data', 'epochs', '{}_{}_avg_16s-epo.fif' .format(subj, c))
        epochs = mne.read_epochs(file)
        picks = mne.pick_types(epochs.info, eeg=True, exclude='bads')
        psds, freqs = psd_multitaper(epochs, fmin=1, fmax=120, picks=picks, n_jobs=n_jobs)

        # fmin = (1, 4, 7, 10, 13, 30, 60, 90)
        # fmax = (4, 7, 10, 13, 30, 60, 90, 120)

        fmin = (1, 4, 7, 10, 13, 30, 60, 90)
        fmax = (4, 7, 10, 13, 30, 60, 90, 120)

        fq_dat = list()
        for f_mi, f_ma in zip(fmin, fmax):
            print(f_mi, f_ma)
            fq_mask = (freqs >= f_mi) & (freqs <= f_ma)
            fq_pow = psds[:, :, fq_mask].mean(2)
            fq_dat.append(fq_pow)

        fq_dat_mat = np.stack(fq_dat, axis=2)
        cond_data.append(fq_dat_mat)

    gammas = frequencies[-3:]
    cond_data[0] = cond_data[0][:, :, -3:]
    cond_data[1] = cond_data[1][:, :, -3:]

    pow_fig, axes = plt.subplots(2, len(gammas), sharey='row', sharex='row')
    pow_fig.subplots_adjust(wspace=0)
    p_vals = list()
    for ix, f in enumerate(gammas):
        wake_dat = np.mean(cond_data[0][:, :, ix], 1)
        sleep_dat = np.mean(cond_data[1][:, :, ix], 1)
        axes[0, ix].violinplot([wake_dat, sleep_dat], showmeans=True)
        axes[0, ix].set_yscale('log')
        t_real, t_list, p_permuted = permutation_t_test(wake_dat, sleep_dat, n_perm)
        p_vals.append(p_permuted)
        axes[1, ix].hist(t_list, bins=100, facecolor='black')
        axes[1, ix].vlines(t_real, ymin=0, ymax=400, linestyles='--')
    # epochs.plot(scalings={'eeg': 120e-5}, n_channels=epochs.info['nchan'], n_epochs=1)

    p_vals_corr = multipletests(p_vals, 0.05, 'holm')[1]
    for ax in range(len(gammas)):
        axes[0, ax].set_title(p_vals_corr[ax])

    print(p_vals_corr)
    pow_fig.savefig('{}/{}/figures/{}_gamma_pow_fig.eps' .format(study_path, subj, subj), format='eps', dpi=300)


def make_n_ch_hist():
    subjects = ['s1', 's2', 's3']

    plt.style.use('classic')
    print('pli matrix anatomical')
    lobes_labels = ['Frontal', 'Temporal', 'Parietal', 'Occipital']

    c = 'wake'
    ref = 'avg'
    l = '8s'

    dfs = list()

    for subj in subjects:
        con, con_mat, con_tril, freqs, n_ch, ch_names, pos = load_pli_ieeg(study_path, subj, c, ref, l)

        gm_chans, gm_names, gm_ch_info = select_gm_chans(subj, ref, study_path, ch_names)

        # sort by anatomy
        gm_ch_info['Original_ix'] = np.arange(0, len(gm_ch_info))

        lobes = [x[1] for x in gm_ch_info['Lobe'].str.split().tolist()]
        gm_ch_info['LobePlot'] = pd.Categorical(lobes, lobes_labels)
        gm_ch_info_ordered = gm_ch_info.sort_values(by=['LobePlot', 'Electrode'])

        dfs.append(gm_ch_info_ordered)

    all_ch = pd.concat(dfs, ignore_index=True)
    all_ch.LobePlot.value_counts().plot(kind='bar')

    from collections import Counter
    chans = Counter(all_ch.LobePlot.tolist())
    chans_nr = [chans[k] for k in lobes_labels]

    fig, ax = plt.subplots()
    ax.barh(np.arange(len(lobes_labels)), chans_nr)
    ax.set_yticks(np.arange(len(lobes_labels))+0.5)
    ax.invert_yaxis()
    ax.set_yticklabels(lobes_labels)
    fig.savefig(op.join(study_path, 'figures', 'all_ch_bars.eps'), format='eps', dpi=300)


def all_subj_nodes(subjs):
    conds = ['wake', 'sleep']
    freqs = ['alpha-H', 'gamma-L', 'gamma-M', 'gamma-H']

    for cond in conds:
        for fq in freqs:
            subjs_nodes = list()
            for ix_s, subj in enumerate(subjs):
                s_file = op.join(study_path, subj, 'results', 'node_files', '{}_avg_{}_{}_nodes_avgPLI.node' .format(subj, cond, fq))
                s_nodes = np.loadtxt(s_file)

                # rescale sizes by wpli
                s_nodes[:, 4] = d3_scale(s_nodes[:, 3], out_range=(0, 1.))
                s_nodes[:, 3] = ix_s + 1
                s_nodes[-2, 1] = 40.
                s_nodes = np.vstack([s_nodes, np.array([[-70, 40, 70, ix_s+1, 0], [-70, 70, 70, ix_s+1, 1.]])])

                subjs_nodes.append(s_nodes)
            all_nodes = np.concatenate(subjs_nodes)
            nodes_file = op.join(study_path, 'varios', 'all_subj_nodes_{}_{}_for_glass.node' .format(cond, fq))
            make_bnw_nodes(nodes_file, all_nodes[:, 0:3], all_nodes[:, 3], all_nodes[:, 4])


if __name__ == '__main__':
    for s in ['s1', 's2', 's3']:
        for r in ['avg']:
            print('Subject: {}' .format(s))

            # plot_pli_violin_all(s)
            # plot_pli_violin(s, r, '8s')
            # plot_pli_violin_gamma(s, r, '8s')
            # plot_matrix_pli(s, conditions, r, '8s')
            # pli_by_distance(s, r, '8s')
            # pli_by_distance_gamma(s, r, '8s')
            # plot_smi_violin_all(s)
            # plot_smi_violin(s, r, '500ms')
            # plot_matrix_smi(s, conditions, r, '500ms')
            plot_matrix_pli_anotomical(s, conditions, r, '8s')
        # plot_pli_matrix_scalp(s)
        # create_node_file(s, r)
        # for c in conditions:
            # create_edge_file(s, r, c)
            # create_graph_nodes(s, r, c)
            # plot_da(s)
        #export_deg_table(s, r)
        # conn_hfo_analysis(s)
        #power_analysis(s)

