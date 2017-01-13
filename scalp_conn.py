import mne
from wake_sleep_info import study_path, bad_channels_scalp, scalings

subj = 's5'
conds = ['wake', 'sleep']
c = 'wake'

for c in conds:
    fname = '{}/{}/data/raw/{}_EGI_{}.set' .format(study_path, subj, subj, c)
    raw = mne.io.read_raw_eeglab(fname, preload=True)
    bads_nr = bad_channels_scalp[subj][c]
    bads = ['E' + str(s) for s in bads_nr]
    # mne.viz.plot_sensors(raw.info, kind='3d')
    raw.info['bads'] = bads.copy()
    montage = mne.channels.read_montage('GSN-HydroCel-256')
    fidz = mne.io.read_fiducials()
    raw.set_montage(montage)

    raw.plot(scalings={'eeg': 150e-7}, n_channels=128)
    mne.gui.coregistration()

raw_test = mne.io.read_raw_edf('{}/{}/data/raw/egi_fid_test.edf' .format(study_path, subj))
