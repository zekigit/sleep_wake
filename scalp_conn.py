import mne
from wake_sleep_info import study_path, bad_channels_scalp, n_jobs
from ieeg_fx import add_event_to_continuous
import numpy as np

subj = 's5'
conds = ['wake', 'sleep']
c = 'wake'
use_CSD = True

for c in conds:
    fname = '{}/{}/data/raw/{}_EGI_{}.set' .format(study_path, subj, subj, c)
    raw = mne.io.read_raw_eeglab(fname, preload=True)
    bads_nr = bad_channels_scalp[subj][c]
    bads = ['E' + str(s) for s in bads_nr]
    # mne.viz.plot_sensors(raw.info, kind='3d')
    raw.info['bads'] = bads.copy()
    montage = mne.channels.read_montage('GSN-HydroCel-256')
    raw.set_montage(montage)

    # raw.plot(scalings={'eeg': 15e-6}, n_channels=128)
    # fidz = mne.io.read_fiducials()
    # mne.gui.coregistration()

    raw, ref = mne.io.set_eeg_reference(raw, ref_channels=None)
    raw.apply_proj()

    raw.filter(l_freq=0.1, h_freq=40, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=n_jobs,
               phase='zero', fir_window="hamming")
    # raw.plot(n_channels=32, duration=5)

    picks_eeg = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, exclude='bads')
    reject = {'eeg': 60e-5}
    # ICA
    n_components = 20
    method = 'infomax'
    ica = mne.preprocessing.ICA(n_components=n_components, method=method)
    ica.fit(raw, picks=picks_eeg, decim=None, reject=reject)
    print(ica)
    ica.plot_components()
    comp_to_delete = [int(x) for x in input('Candidates to delete (e.g.: 0 1 3; if None, press Enter): ').split()]

    if comp_to_delete:
        ica.plot_properties(raw, picks=comp_to_delete, psd_args={'fmax': 40.})
        delete = input('Confirm deletion (y/n or list to delete): ')
        if delete == 'y':
            ica.exclude = comp_to_delete.copy()
            raw = ica.apply(raw)
        elif isinstance(delete, list):
            comp_to_delete = delete.copy()
            ica.exclude = comp_to_delete.copy()
            raw = ica.apply(raw)
        elif delete == 'n':
            pass

    raw.interpolate_bads()

    t_min = 0
    t_max = 8

    events = add_event_to_continuous(raw, t_max)
    epochs = mne.Epochs(raw, events=events, event_id={str(t_max): 1}, tmin=t_min, tmax=t_max,
                        baseline=(None, None), reject=reject, preload=True, add_eeg_ref=False)
    epochs.info['len'] = t_max
    # epochs.plot(scalings={'eeg': 150e-5}, n_channels=128)
    fmin = (1, 4, 7, 10, 13, 30)
    fmax = (4, 7, 10, 13, 30, 40)

    # print('Calculating Connectivity')
    # con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs, method='wpli2_debiased',
    #                                                                                sfreq=epochs.info['sfreq'],
    #                                                                                mode='multitaper', fmin=fmin,
    #                                                                                fmax=fmax, faverage=True,
    #                                                                                mt_adaptive=False, n_jobs=n_jobs,
    #                                                                                verbose=False)
    # con_mat = np.copy(con)
    #
    # upper = np.triu_indices_from(con_mat[:, :, 0])
    # for fq in range(len(freqs)):
    #     swap = np.swapaxes(con_mat[:, :, fq], 0, 1)
    #     for val in zip(upper[0], upper[1]):
    #         con_mat[:, :, fq][val] = swap[val]
    #
    # ch_names = epochs.info['ch_names']
    # pli_results = dict(con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names,
    #                    subj=subj, cond=c, ref='avg', len=t_max)
    #
    # print('saving')
    # folder = '{}/{}/results/pli' .format(study_path, subj)
    # np.savez('{}/{}_{}_{}s_pli_scalp'.format(folder, pli_results['subj'], pli_results['cond'], pli_results['len']),
    #          con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names,
    #          subj=pli_results['subj'], cond=pli_results['cond'], ref=pli_results['ref'], len=pli_results['len'])
    epochs.resample(500)
    epochs.save('{}/{}/data/epochs/{}_EGI_{}-epo.fif' .format(study_path, subj, subj, c))


