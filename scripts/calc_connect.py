import mne
from wake_sleep_info import subjects, conditions, dat_path, results_path
from connectivity_fxs import add_event_to_continuous
import os.path as op
import numpy as np


def calc_connectivity(file, epoch_length):
    save = True
    raw = mne.io.read_raw_eeglab(file, preload=True)
    # raw.plot(scalings={'eeg': 100e-6}, n_channels=raw.info['nchan'])
    raw, ref = mne.io.set_eeg_reference(raw, ref_channels=None)
    raw, events = add_event_to_continuous(raw, epoch_length)
    # raw.plot(events=events, duration=15, scalings={'eeg': 100e-6}, n_channels=raw.info['nchan'])
    picks = mne.pick_types(raw.info, eeg=True)
    reject = {'eeg': 150e6}
    epochs = mne.Epochs(raw, events, event_id={'10s': 1}, tmin=0, tmax=10, baseline=(None, None),
                        picks=picks, reject=reject)
    epochs.drop_bad()
    fmin = (1, 4, 8, 13, 60)
    fmax = (4, 8, 13, 30, 120)
    con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs, method='wpli',
                                                                                   sfreq=epochs.info['sfreq'],
                                                                                   mode='multitaper', fmin=fmin,
                                                                                   fmax=fmax, faverage=True,
                                                                                   mt_adaptive=False, n_jobs=2)

    con_mat = np.copy(con)

    upper = np.triu_indices_from(con_mat[:, :, 0])
    for fq in range(len(freqs)):
        swap = np.swapaxes(con_mat[:, :, fq], 0, 1)
        for val in zip(upper[0], upper[1]):
            con_mat[:, :, fq][val] = swap[val]

    ch_names = epochs.info['ch_names']

    if save:
        outfile = results_path + 'connectivity_{}_{}'.format(subj, cond)
        np.savez(outfile, con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names)

    return con, freqs, times, n_epochs, n_tapers, con_mat, ch_names


subj = 'P14'

for subj in subjects:
    for cond in conditions[subj]:
        archivo = '{}_{}_OK.set' .format(subj, cond)
        file = op.join(dat_path, 'datos_v1', archivo)

        con, freqs, times, n_epochs, n_tapers, con_mat, ch_names = calc_connectivity(file, epoch_length=10)
