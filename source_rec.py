import mne
from wake_sleep_info import study_path, n_jobs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
import numpy as np
import matplotlib.pyplot as plt

subject = 's5'
subjects_dir = '/Applications/freesurfer/subjects'

reject = {'eeg': 60e-5}


fif_file = '{}/{}/data/raw/{}_EGI_wake-raw.fif' .format(study_path, subject, subject, subject)
epochs_file = '{}/{}/data/epochs/{}_EGI_wake-epo.fif' .format(study_path, subject, subject, subject)

fwd_file = '{}/{}/info/{}-fwd.fif' .format(study_path, subject, subject)
cov_file = '{}/{}/info/{}_EGI-cov.fif' .format(study_path, subject, subject)
inv_file = '{}/{}/info/{}_EGI-inv.fif' .format(study_path, subject, subject)

raw = mne.io.read_raw_fif(fif_file)
# raw.plot(n_channels=256)

##  Covariance
# cov = mne.compute_raw_covariance(raw, method='auto', tmin=30, tmax=50, reject=reject)
# cov_file = '{}/{}/info/{}_EGI-cov.fif' .format(study_path, subject, subject)
# mne.write_cov(cov_file, cov)

fwd = mne.read_forward_solution(fwd_file)
cov = mne.read_cov(cov_file)
# inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2)
# mne.minimum_norm.write_inverse_operator(inv_file, inv)
inv = read_inverse_operator(inv_file)

epochs = mne.read_epochs(epochs_file, preload=True)

snr = 1.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'
stcs = apply_inverse_epochs(epochs, inv, lambda2, method, pick_ori='normal', return_generator=True)

labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)

label_colors = [label.color for label in labels]
src = inv['src']

label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip', return_generator=True)

fmin = (1, 4, 7, 10, 13, 30)
fmax = (4, 7, 10, 13, 30, 40)

print('Calculating Connectivity')
con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(label_ts, method='wpli2_debiased',
                                                                               sfreq=epochs.info['sfreq'],
                                                                               mode='multitaper', fmin=fmin,
                                                                               fmax=fmax, faverage=True,
                                                                               mt_adaptive=True, n_jobs=n_jobs,
                                                                               verbose=True)


label_names = [label.name for label in labels]

lh_labels = [name for name in label_names if name.endswith('lh')]
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

rh_labels = [label[:-2] for (yp, label) in sorted(zip(label_ypos, lh_labels))]
node_order = list()
node_order.extend(lh_labels[::-1])
# node_order.extend(rh_labels)
rh_labels_corr = [label + 'rh' for label in rh_labels]
node_order.extend((rh_labels_corr))

fqs = ['delta', 'theta', 'alpha l', 'alpha h', 'beta', 'gamma']
fq = 3
# del label_names[label_names.index('bankssts-rh')]
node_angles = circular_layout(label_names, node_order, start_pos=90, group_boundaries=[0, len(label_names) / 2])

for fq in range(len(fqs)):
    plot_connectivity_circle(con[:, :, fq], label_names, n_lines=300, node_angles=node_angles, node_colors=label_colors, title='dwPLI ' + fqs[fq],
                             vmin=0, vmax=1)
    fig_file = '{}/{}/figures/source_dwpli_{}.eps' .format(study_path, subject, fqs[fq])
    plt.savefig(fig_file, facecolor='black')
