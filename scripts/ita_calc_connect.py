import matplotlib.pyplot as plt
import numpy as np
from connectivity_fxs import add_event_to_continuous, create_con_mat
from ita_info import bad_ch_eeg, results_path
import mne

cond = 'wake'
save = True

archivo = '/Users/lpen/PycharmProjects/Wake_Sleep/milano/data/set/' + cond + '_EGI.set'

montage = mne.channels.read_montage('GSN-HydroCel-256')

raw = mne.io.read_raw_eeglab(archivo, montage=montage, preload=True)
# mne.viz.plot_raw_psd(raw)

raw.filter(l_freq=1, h_freq=40)
# mne.viz.plot_raw_psd(raw, fmax=35)
raw, ref = mne.io.set_eeg_reference(raw, ref_channels=None)

bad_ch = bad_ch_eeg[cond]
raw.info['bads'] = bad_ch
picks_eeg = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, exclude='bads')
print('' * 2)

reject = {'eeg': 200e-6}

# # ICA
# n_components = 20
# method = 'infomax'
# decim = 4
# ica = mne.preprocessing.ICA(n_components=n_components, method=method)
# ica.fit(raw, picks=picks_eeg, decim=decim, reject=reject)
# print(ica)
# ica.plot_components()
# comp_to_delete = [int(x) for x in input('Components to delete (ej: 1 2 3): ').split()]
# if comp_to_delete:
#     ica.exclude = comp_to_delete
#     raw = ica.apply(raw)
#

raw.interpolate_bads()
# raw.plot(n_channels=256, duration=60, scalings={'eeg': 50e-6})
picks_eeg_all = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, exclude='bads')

raw, events = add_event_to_continuous(raw, 10)

epochs = mne.Epochs(raw, events, event_id={'10s': 1}, tmin=0, tmax=10, baseline=(None, None),
                    picks=picks_eeg_all, reject=reject)

print('Original nr of epochs: {}' .format(len(epochs.events)))
epochs.drop_bad()
epochs.plot_drop_log()
print('New nr of epochs: {}' .format(len(epochs.events)))


ch_names = epochs.info['ch_names']

fmin = (1, 4, 8, 10, 13)
fmax = (4, 8, 10, 13, 30)

print('' * 3)
print('calculating connectivity')
con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs, method='wpli',
                                                                               sfreq=epochs.info['sfreq'],
                                                                               mode='multitaper', fmin=fmin,
                                                                               fmax=fmax, faverage=True,
                                                                               mt_adaptive=False, n_jobs=2)

con_mat = create_con_mat(con, freqs)

if save:
    outfile = results_path + 'connectivity_EGI_{}'.format(cond)
    np.savez(outfile, con_tril=con, freqs=freqs, n_epochs=n_epochs, n_tapers=n_tapers, con_mat=con_mat, ch_names=ch_names)

