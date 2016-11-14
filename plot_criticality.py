import numpy as np
import matplotlib.pyplot as plt

from wake_sleep_info import subjects, conditions, study_path

hists, axes = plt.subplots(1, 3, figsize=(12, 6))
for ix, subj in enumerate(subjects):
    wake_results = np.load('dfa_datos{}_wakehilbert.npy'.format(subjects[ix]))
    sleep_results = np.load('dfa_datos{}_sleephilbert.npy'.format(subjects[ix]))
    if subj == 1:
        wake_results = np.delete(wake_results, -1, 0)
        sleep_results = np.delete(sleep_results, -1, 0)
    wake_alpha = wake_results[:, 2]
    sleep_alpha = sleep_results[:, 2]
    # fig, ax = plt.subplots()
    axes[ix].hist([wake_alpha, sleep_alpha],  label=['wake', 'sleep'], color=['red', 'blue'])
    axes[ix].set(title=subj, xlabel='DFA Alpha')

axes[0].set_ylabel('Channels')
axes[0].legend(loc=2)

hists.savefig('dfa_alpha_hist_3suj.pdf', format='pdf', dpi=300)
    # ax.hist(sleep_alpha, n_bins)
#