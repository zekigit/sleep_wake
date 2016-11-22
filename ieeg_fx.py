import pandas as pd
import numpy as np


def check_channels(ch_info, raw):
    indexes = list()
    for ix, ch in ch_info.iterrows():
        if ch['Electrode'] in raw.info['ch_names']:
            indexes.append(ix)
    ch_info_new = ch_info.loc[indexes]
    return ch_info_new


def make_bip_chans(ch_info):
    bip_info = {'Nr': [], 'Name': [], 'Electrode': [], 'White Grey': [], 'Area Specific': [], 'Area MNI Mango': [],
                'natX': [], 'natY': [], 'natZ': [],
                'normX': [], 'normY': [], 'normZ': [], 'Subj': []}
    anodes = list()
    cathodes = list()

    for ix, ch in ch_info.iterrows():
        if (ix < len(ch_info)-1) and (ch_info['Nr'].iloc[ix+1] == ch_info['Nr'].iloc[ix]+1) and \
                (ch_info['Name'].iloc[ix] == ch_info['Name'].iloc[ix+1]) and (ch_info['White Grey'].iloc[ix] == ch_info['White Grey'].iloc[ix+1]):
            for column in bip_info.keys():
                col = ch_info[column].iloc[ix]
                if column in ('natX', 'natY', 'natZ', 'normX', 'normY', 'normZ'):
                    col = np.average([ch_info[column].iloc[ix], ch_info[column].iloc[ix+1]])
                if column == 'Electrode':
                    anodes.append(col), cathodes.append(ch_info['Electrode'].iloc[ix+1])
                    col = '{}-{}' .format(col, cathodes[-1] )
                bip_info[column].append(col)
    bip_chans = pd.DataFrame(bip_info)
    print('nr anodes:{} nr cathodes:{}' .format(len(anodes), len(cathodes)))
    return bip_chans, anodes, cathodes


def add_event_to_continuous(raw, epoch_length):
    w_length = raw.info['sfreq']*epoch_length
    ev_times = [0]
    while ev_times[-1]+w_length < raw.last_samp:
        ev_times.extend([ev_times[-1] + w_length])
    ev_samples = np.array(ev_times, dtype=int)
    zeros = np.zeros(len(ev_samples), dtype=int)
    marks = np.repeat(int(1), len(ev_samples))
    events = np.vstack((ev_samples, zeros, marks))
    events = np.transpose(events)
    # raw.add_events(events)
    return events


