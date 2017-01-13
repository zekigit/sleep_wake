import mne
import matplotlib.pyplot as plt
from wake_sleep_info import subjects, conditions, study_path, scalings, lengths_nr
from ieeg_fx import read_ch_info, make_bip_chans, check_channels, ch_info_coords
from study_fx import prepro, cut_epochs, export_fif_mlab, raw_and_powerplots, calc_pli
import os
import pandas as pd

"""
Requires the following folder structure:
study / subj / info, data/raw data/epochs, results/smi results/pli, figures
Also "subjID_ch_info.xls" in info folder with channel information
"""
# Adjust display
pd.set_option('display.expand_frame_repr', False)
os.chdir(study_path)


def ieeg_main(subj, cond):
    data_file = '{}/{}/data/raw/{}_SEEG_{}.set' .format(study_path, subj, subj, cond)
    fig_path = '{}/{}/figures' .format(study_path, subj)
    res_path = '{}/{}/results' .format(study_path, subj)

    raw = mne.io.read_raw_eeglab(data_file, preload=True)
    raw.info['cond'] = cond
    raw.info['subj'] = subj
    # plt.show()
    ch_info_all = ch_info_coords(subj, study_path)
    ch_info_ok = check_channels(ch_info_all, raw)
    bip_info, anodes, cathodes = make_bip_chans(ch_info_ok)

    bads = ch_info_ok['Electrode'][ch_info_ok['White Grey'] != 'Grey Matter'].tolist()

    raw.info['bads'] = [ch for ch in bads if ch in raw.info['ch_names']]
    # raw.plot(scalings={'eeg': 150e-6}, n_channels=raw.info['nchan'])

    raw_avg, raw_bip = prepro(raw, bip_info, anodes=anodes, cathodes=cathodes)

    raw_bip.info['bads'] = bip_info['Electrode'][bip_info['White Grey'] != 'Grey Matter'].tolist()

    for r in [raw_bip, raw_avg]:
        raw_and_powerplots(r, scalings, fig_path)

    epo_avg = cut_epochs(raw_avg)
    epo_bip = cut_epochs(raw_bip)

    ch_info_ok.to_pickle(os.path.join(study_path, subj, 'info', '{}_avg_info_coords.pkl'.format(subj)))
    bip_info.to_pickle(os.path.join(study_path, subj, 'info', '{}_bip_info_coords.pkl'.format(subj)))

    for epo in epo_avg + epo_bip:
        pli_results = calc_pli(epo, lengths_nr=lengths_nr, folder='{}/pli' .format(res_path))
        export_fif_mlab(epo, subj, folder='{}/{}/data/epochs' .format(study_path, subj))
    return

conditions = ['wake', 'sleep']

if __name__ == '__main__':
    for c in conditions:
        ieeg_main('s5', c)
