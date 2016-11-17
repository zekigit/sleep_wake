import mne
import pandas as pd
import numpy as np
from ieeg_fx import add_event_to_continuous


def read_ch_info(subj, study_path):
    ch_info = pd.read_excel('{}/{}/info/{}_ch_info.xlsx' .format(study_path, subj, subj))
    # channels = ch_info['Name'].map(str) + ch_info['Nr'].astype(str)
    ch_info['Electrode'] = ch_info.Name.str.cat(ch_info.Nr.astype(str))
    ch_info['Subj'] = subj
    return ch_info


def prepro(raw, new_info, anodes, cathodes):
    # raw.plot(scalings={'eeg': 100e-6}, n_channels=raw.info['nchan'])
    raw_avg, ref = mne.io.set_eeg_reference(raw, ref_channels=None)
    raw_bip = mne.io.set_bipolar_reference(raw, anode=anodes, cathode=cathodes)
    # raw.plot(events=events, duration=15, scalings={'eeg': 100e-6}, n_channels=raw.info['nchan'])
    picks = mne.pick_types(raw_avg.info, eeg=True)
    return raw_avg, raw_bip


def power_estimates(data):
    pass


def cut_epochs(data):
    reject = {'eeg': 200}
    tmin = 0
    tmaxs = [0.5, 1, 5, 10]
    epochs_list = list()
    for m in tmaxs:
        raw, events = add_event_to_continuous(data, m)
        epochs_list.append(mne.Epochs(raw, events=events, event_id={str(m): 1}, tmin=tmin, tmax=m,
                                      baseline=(None, None), reject=reject, preload=True))
    return epochs_list


def prepare_for_da(epochs):
    # run_da_from_py
    pass


def prepare_smi(epochs):
    # run_smi_from_py
    pass


def calc_ati(epochs):
    pass


def calc_criticality(data):
    pass


