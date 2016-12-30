import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from ieeg_fx import add_event_to_continuous
import scipy.io as scio


def prepro(raw, new_info, anodes, cathodes):
    # raw.plot(scalings={'eeg': 100e-6}, n_channels=raw.info['nchan'])
    base = raw.copy()  # Report mne! (also del bip chans)
    raw_avg, ref = mne.io.set_eeg_reference(base, ref_channels=None, copy=True)
    raw_avg.apply_proj()
    raw_avg.info['ref'] = 'avg'
    raw.info['bads'] = []
    raw_bip = mne.io.set_bipolar_reference(raw, anode=anodes, cathode=cathodes)
    raw_bip.info['ref'] = 'bip'
    return raw_avg, raw_bip


def raw_and_powerplots(raw, scalings, fig_path):
    # raw.plot(scalings=scalings, n_channels=raw.info['nchan'])
    # plt.show()
    # input('Press Enter to continue')

    power_raw = raw.plot_psd(show=False, fmax=140)
    power_raw.savefig('{}/{}_{}_{}_power' .format(fig_path, raw.info['subj'], raw.info['cond'], raw.info['ref']))
    pass


def cut_epochs(data):
    reject = {'eeg': 200}
    tmin = 0
    tmaxs = [16]
    epochs_list = list()
    for m in tmaxs:
        events = add_event_to_continuous(data, m)
        epochs = mne.Epochs(data, events=events, event_id={str(m): 1}, tmin=tmin, tmax=m,
                            baseline=(None, None), reject=reject, preload=True, add_eeg_ref=False)
        epochs.info['len'] = list(epochs.event_id.keys())[0]
        epochs_list.append(epochs)
    return epochs_list


def export_fif_mlab(epochs, subj, folder):
    length = list(epochs.event_id.keys())[0]
    cond = epochs.info['cond']
    if len(epochs.info['projs']):
        ref = 'avg'
    else:
        ref = 'bip'

    exp_data = dict(length=length,
                    data=epochs.get_data(),
                    sfreq=epochs.info['sfreq'],
                    subj=subj,
                    ref=ref,
                    ch_names=epochs.info['ch_names'])

    if length == '0.5':
        length = '500m'
    elif length == '0.25':
        length = '250m'

    file = '{}/{}_{}_{}_{}s' . format(folder, subj, cond, ref, length)
    print('saving ' + file)
    scio.savemat(file, exp_data)
    epochs.save(file + '-epo.fif')


def calc_pli(epochs, lengths_nr, folder):
    fmin = (1, 4, 7, 10, 13, 30, 60, 110)
    fmax = (4, 7, 10, 13, 30, 48, 90, 140)
    epo_cat = epochs.copy()

    for l in reversed(lengths_nr):
        if l != 16:
            epo1 = epo_cat.copy()
            epo2 = epo_cat.copy()
            half1 = epo1.crop(tmin=None, tmax=l)
            half2 = epo2.crop(tmin=l, tmax=None)
            half2.times = half1.times.copy()
            epo_cat = mne.concatenate_epochs([half1, half2])

        if l == 8:
            print('computing wpli - subj: %s - cond: %s - ref: %s' % (epochs.info['subj'], epochs.info['cond'], epochs.info['ref']))
            print('len: {}  -  n trials: {}' .format(l, len(epo_cat)))

            if l == 0.25:
                fmin = (7, 10, 13, 30, 60, 110)
                fmax = (10, 13, 30, 48, 90, 140)

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

            ch_names = epochs.info['ch_names']
            pli_results = dict(con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names,
                               subj=epochs.info['subj'], cond=epochs.info['cond'], ref=epochs.info['ref'], len=l)

            print('saving')
            np.savez('{}/{}_{}_{}_{}s_pli' .format(folder, pli_results['subj'], pli_results['cond'], pli_results['ref'], pli_results['len']),
                     con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names,
                     subj=epochs.info['subj'], cond=epochs.info['cond'], ref=epochs.info['ref'], len=l)


def plot_connectivity(pli_results):
    con = pli_results['con_tril']
    con_mat = pli_results['con_mat']
    freqs = len(pli_results['freqs'])

    # PLOT
    titles = ['delta (1-4hz)', 'theta (4-7hz)', 'alpha l (7-10hz)', 'alpha h (10-13hz)', 'beta (13-30hz)',
              'gamma l (30-48hz)', 'gamma m (60-90hz)', 'gamma h (110-140hz)']

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
        # ax.set_title(titles[idx])

    cb = con_fig.colorbar(im, cax=grid.cbar_axes[0])
    cb.ax.set_title('wPLI', loc='right')


def crop_and_order_epochs(epochs):
    length = epochs.info['len']
    epo = epochs.copy()
    half1 = epo.crop(tmin=0, tmax=length/2).copy()
    half2 = epo.crop(tmin=length/2, tmax=length)

    pass


def calc_ati(epochs):
    pass


def calc_criticality(data):
    pass


