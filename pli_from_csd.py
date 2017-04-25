import mne
from connectivity_fxs import loadmat
from wake_sleep_info import study_path, conditions, n_jobs
import numpy as np
import scipy.io as scio

subj = 's5'

for c in conditions:
    fname = '{}/{}/data/epochs/{}_EGI_{}-epo.fif' .format(study_path, subj, subj, c)
    epochs = mne.read_epochs(fname, preload=True)
    epochs.drop_channels(['STI 014'])
    csd_load = loadmat('{}/{}/data/epochs/{}_{}_CSD.mat' .format(study_path, subj, subj, c))
    csd = csd_load['csd_data']
    csd = np.transpose(csd, (2, 0, 1))
    epochs._data = csd.copy()

    t_min = 0
    t_max = 8

    fmin = (1, 4, 7, 10, 13, 30)
    fmax = (4, 7, 10, 13, 30, 40)

    print('Calculating Connectivity')
    con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs, method='wpli2_debiased',
                                                                                   sfreq=epochs.info['sfreq'],
                                                                                   mode='multitaper', fmin=fmin,
                                                                                   fmax=fmax, faverage=True,
                                                                                   mt_adaptive=False, n_jobs=n_jobs,
                                                                                   verbose=False)
    con_mat = np.copy(con)

    upper = np.triu_indices_from(con_mat[:, :, 0])
    for fq in range(len(freqs)):
        swap = np.swapaxes(con_mat[:, :, fq], 0, 1)
        for val in zip(upper[0], upper[1]):
            con_mat[:, :, fq][val] = swap[val]

    ch_names = epochs.info['ch_names']
    pli_results = dict(con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names,
                       subj=subj, cond=c, ref='csd', len=t_max)

    print('saving')
    folder = '{}/{}/results/pli' .format(study_path, subj)
    np.savez('{}/{}_{}_{}s_pli_scalp_CSD'.format(folder, pli_results['subj'], pli_results['cond'], pli_results['len']),
             con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names,
             subj=pli_results['subj'], cond=pli_results['cond'], ref=pli_results['ref'], len=pli_results['len'])

    exp_data = dict(con=con,
                    con_mat=con_mat,
                    con_tril=con,
                    freqs=freqs,
                    ch_names=ch_names)

    file = '{}/{}/results/pli/{}_{}_pli_CSD.mat' .format(study_path, subj, subj, c)
    print('saving ' + file)
    scio.savemat(file, exp_data)
