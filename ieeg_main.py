import mne
import matplotlib.pyplot as plt
from wake_sleep_info import subjects, conditions, study_path, results_path, data_path, scalings, figures_path
from ieeg_fx import make_bip_chans, check_channels
# from connectivity_fxs import add_event_to_continuous
from study_fx import read_ch_info, prepro
import os
import pandas as pd

"""
Requires the following folder structure:
study / subj / info, data, results, figures
Also "s1.xls" in info folder with channel information
"""
# Adjust display
pd.set_option('display.expand_frame_repr', False)
os.chdir(study_path)


def ieeg_main(subj, cond):
    data_file = '{}/{}/data/{}_{}.set' .format(study_path, subj, subj, cond)
    raw = mne.io.read_raw_eeglab(data_file, preload=True)

    ch_info_all = read_ch_info(subj, study_path)
    ch_info_ok = check_channels(ch_info_all, raw)
    bip_info, anodes, cathodes = make_bip_chans(ch_info_ok)
    raw_avg, raw_bip = prepro(raw, bip_info, anodes=anodes, cathodes=cathodes)

    raw_bip.info['bads'] = bip_info['Electrode'][bip_info['White Grey'] != 'Grey Matter'].tolist()
    raw_avg.info['bads'] = ch_info_ok['Electrode'][ch_info_ok['White Grey'] != 'Grey Matter'].tolist()

    raw_avg.plot(scalings=scalings, n_channels=raw_avg.info['nchan'])
    plt.show()
    input('Press Enter to continue')
    raw_bip.plot(scalings=scalings, n_channels=raw_bip.info['nchan'])
    plt.show()
    input('Press Enter to continue')

    power_raw_avg = raw_avg.plot_psd(show=False, fmax=140)
    power_raw_avg.savefig(study_path + '/{}/figures/{}_{}_power_avg.svg' .format(subj, subj, cond))
    power_raw_bip = raw_bip.plot_psd(show=False, fmax=140)
    power_raw_bip.savefig(study_path + '/{}/figures/{}_{}_power_bip.svg' .format(subj, subj, cond))


    # epochs_avg = mne.Epochs(raw_avg, )
    # epochs_bip = mne.Epochs(raw_bip)

    return ch_info_all, raw_avg, raw_bip


subjects = ['s2']

for s in subjects:
    for c in conditions:
        ch_information, prep_avg, prep_bip = ieeg_main(s, c)

