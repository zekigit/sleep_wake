import mne
import matplotlib.pyplot as plt
import numpy as np
from wake_sleep_info import study_path, conditions, frequencies
from ieeg_fx import select_gm_chans, ch_info_coords, check_channels, make_bip_chans, load_pli_ieeg
from mpl_toolkits.axes_grid1 import ImageGrid
from study_fx import prepro, cut_epochs, calc_pli
from connectivity_fxs import randomize_epochs_phase, binarize_mat, calc_graph_metrics
import pandas as pd
import os.path as op


def shuff_pli(subj):
    ref = 'avg'
    l = 8
    for cond in conditions:
        data_file = '{}/{}/data/raw/{}_{}.set'.format(study_path, subj, subj, cond)
        fig_path = '{}/{}/figures'.format(study_path, subj)
        res_path = '{}/{}/results'.format(study_path, subj)

        raw = mne.io.read_raw_eeglab(data_file, preload=True)
        raw.info['cond'] = cond
        raw.info['subj'] = subj
        # plt.show()
        ch_info_all = ch_info_coords(subj, study_path)
        ch_info_ok = check_channels(ch_info_all, raw)
        bip_info, anodes, cathodes = make_bip_chans(ch_info_ok)

        bads = ch_info_ok['Electrode'][ch_info_ok['White Grey'] != 'Grey Matter'].tolist()

        raw.info['bads'] = [ch for ch in bads if ch in raw.info['ch_names']]
        # raw.plot(scalings={'eeg': 150e-6}, n_channels=raw.info['nchan'])

        raw_avg, raw_bip = prepro(raw, bip_info, anodes=anodes, cathodes=cathodes)

        epo_avg = cut_epochs(raw_avg)

        epo_cat = epo_avg[0].copy()
        epo1 = epo_cat.copy()
        epo2 = epo_cat.copy()
        half1 = epo1.crop(tmin=None, tmax=l)
        half2 = epo2.crop(tmin=l, tmax=None)
        half2.times = half1.times.copy()
        epo_cat = mne.concatenate_epochs([half1, half2])
        epo_cat = randomize_epochs_phase(epo_cat)

        fmin = (1, 4, 7, 10, 13, 30, 60, 110)
        fmax = (4, 7, 10, 13, 30, 48, 90, 140)

        con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epo_cat, method='wpli',
                                                                                       sfreq=epo_cat.info['sfreq'],
                                                                                       mode='multitaper', fmin=fmin,
                                                                                       fmax=fmax, faverage=True,
                                                                                       mt_adaptive=False, n_jobs=2,
                                                                                       verbose=False)
        con_mat = np.copy(con)

        upper = np.triu_indices_from(con_mat[:, :, 0])
        for fq in range(len(freqs)):
            swap = np.swapaxes(con_mat[:, :, fq], 0, 1)
            for val in zip(upper[0], upper[1]):
                con_mat[:, :, fq][val] = swap[val]

        ch_names = epo_cat.info['ch_names']
        pli_results = dict(con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names,
                           subj=epo_cat.info['subj'], cond=epo_cat.info['cond'], ref=epo_cat.info['ref'], len=l)

        print('saving')
        folder = '{}/{}/results/pli' .format(study_path, subj)
        np.savez('{}/{}_{}_{}_{}s_pli_shuffle'.format(folder, pli_results['subj'], pli_results['cond'], pli_results['ref'], pli_results['len']),
                 con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names,
                 subj=epo_cat.info['subj'], cond=epo_cat.info['cond'], ref=epo_cat.info['ref'], len=l)


def make_shuf_table(subj):
    ref = 'avg'
    l = '8s'
    results = list()
    values = list()
    for c in conditions:
        filename = '{}/{}/results/pli/{}_{}_{}_{}_pli_shuffle.npz'.format(study_path, subj, subj, c, ref, l)
        data = np.load(filename)

        con = data['con_tril'][:-1, :-1, :]
        con_mat = data['con_mat'][:-1, :-1, :]
        freqs = len(data['freqs'])
        n_ch = len(data['ch_names']) - 1
        ch_names = data['ch_names'][:-1]

        con = con[np.tril_indices(n_ch, k=-1)]

        results.append(con_mat)
        values.append(con)

    gm_chans, gm_names, gm_ch_info = select_gm_chans(subj, ref, study_path, ch_names)
    n_gm_chans = len(gm_chans)

    thresholds = list()
    for ix, f in enumerate(frequencies):
        fq_vals = np.concatenate((values[0][gm_chans, ix], values[1][gm_chans, ix]))
        median = np.median(fq_vals)
        sd = np.std(fq_vals)
        threshold = median
        thresholds.append(threshold)

    bin_mats = np.empty((n_gm_chans, n_gm_chans, freqs, len(conditions)))
    conds = list()
    fqs = list()
    chans = list()
    degrees = list()

    for ix_c, c in enumerate(conditions):
        for ix_f, f in enumerate(frequencies):
            mat = results[ix_c][:, :, ix_f]
            bin_mat = binarize_mat(mat[gm_chans[:, None], gm_chans], thresholds[ix_f])
            clustering, degree, graph = calc_graph_metrics(bin_mat, gm_names)
            bin_mats[:, :, ix_f, ix_c] = bin_mat.copy()
            degree_norm = np.array(list(degree.values()))/len(degree)
            degrees.extend(list(degree_norm))
            chans.extend(gm_names)
            fqs.extend([f] * len(degree))
            conds.extend([c] * len(degree))

    degree_df = pd.DataFrame({'Condition': conds, 'Frequency': fqs, 'Channel': chans, 'Degree': degrees, 'Subject': subj})
    save_path = op.join(study_path, subj, 'results', 'tables', '{}_degree_table_shuffle.csv' .format(subj))
    degree_df.to_csv(save_path)


def plot_shuffle(subj):
    ref = 'avg'
    l = '8s'
    for c in conditions:
        filename = '{}/{}/results/pli/{}_{}_{}_{}_pli_shuffle.npz'.format(study_path, subj, subj, c, ref, l)
        data = np.load(filename)
        con_mat = data['con_mat'][:, :, :]
        ch_names = data['ch_names']

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
        con_fig.savefig('{}/{}/figures/{}_{}_{}_pli_{}_conmat_shuffle.eps' .format(study_path, subj, subj, c, ref, l), format='eps', dpi=300)


if __name__ == '__main__':
    for s in ['s1', 's2', 's3']:
        # shuff_pli(s)
        make_shuf_table(s)
