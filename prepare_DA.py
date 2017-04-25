import mne
import scipy.io as scio
from wake_sleep_info import study_path
import os.path as op

conds = ['wake', 'sleep']
ref = 'bip'
subj = 's5'

for c in conds:
    epochs_file = op.join(study_path, subj, 'data', 'epochs', '{}_{}_{}_16s-epo.fif' .format(subj, c, ref))
    epochs = mne.read_epochs(epochs_file, preload=True)
    epochs.resample(512)

    exp_data = dict(length=16,
                    data=epochs.get_data(),
                    sfreq=epochs.info['sfreq'],
                    subj=subj,
                    ref=ref,
                    ch_names=epochs.info['ch_names'])

    file = '{}/{}_{}_{}_{}s' . format(op.join(study_path, subj, 'results'), subj, c, ref, '16')
    print('saving ' + file)
    scio.savemat(file, exp_data)
