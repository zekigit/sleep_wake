import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


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


def read_ch_info(subj, study_path):
    ch_info = pd.read_excel('{}/{}/info/{}_ch_info.xlsx' .format(study_path, subj, subj))
    # channels = ch_info['Name'].map(str) + ch_info['Nr'].astype(str)
    ch_info['Electrode'] = ch_info.Name.str.cat(ch_info.Nr.astype(str))
    ch_info['Subj'] = subj
    return ch_info


def ch_info_coords(subj, study_path):
    markups = pd.read_excel('{}/{}/info/{}_slicer_locs.xlsx'.format(study_path, subj, subj))
    ch_info = read_ch_info(subj, study_path)

    n_elec = list()
    spear = list()
    all_electrodes = dict()

    for ix, e in enumerate(ch_info['Name']):
        if ch_info['Electrode'].iloc[ix] != ch_info['Electrode'].iloc[-1]:
            if e != ch_info['Name'].iloc[ix + 1]:
                n_elec.append(ch_info['Nr'].iloc[ix])
                spear.append(e)
    n_elec.append(ch_info['Nr'].iloc[-1])
    spear.append(ch_info['Name'].iloc[-1])

    for s, n in zip(spear, n_elec):
        e_orig = markups.loc[markups['label'] == '{}1'.format(s)]
        e_end = markups.loc[markups['label'] == s]

        A = np.array((e_orig['x'].iloc[0], e_orig['y'].iloc[0], e_orig['z'].iloc[0]))
        B = np.array((e_end['x'].iloc[0], e_end['y'].iloc[0], e_end['z'].iloc[0]))

        AB = B - A
        length = np.sqrt(AB[0] ** 2 + AB[1] ** 2 + AB[2] ** 2)
        elec_dist = length / n
        AB_norm = AB / length

        # # Alternative Method
        # d = AB[0]
        # new_point = A[0] + elec_dist
        # t = (new_point - A[0]) / AB[0]
        # y = t * AB[1] + A[1]
        # z = t * AB[2] + A[2]

        all_contacts = list()

        for c in range(n - 1):
            contact = A + elec_dist * (1 + c) * AB_norm
            all_contacts.append(contact)
        all_contacts.insert(0, A)
        all_electrodes[s] = all_contacts

    xs = list()
    ys = list()
    zs = list()
    for s in spear:
        contacts = all_electrodes[s]
        for c in contacts:
            xs.append(c[0])
            ys.append(c[1])
            zs.append(c[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs=zs)

    for s in spear:
        contacts = all_electrodes[s]
        for ix, c in enumerate(contacts):
            chan = '{}{}'.format(s, ix + 1)
            if chan in ch_info['Electrode'].values:
                ch_info.set_value(ch_info['Electrode'] == chan, 'natX', c[0])
                ch_info.set_value(ch_info['Electrode'] == chan, 'natY', c[1])
                ch_info.set_value(ch_info['Electrode'] == chan, 'natZ', c[2])
    return ch_info


def load_pli(study_path, subj, cond, ref, win):
    filename = '{}/{}/results/pli/{}_{}_{}_{}_pli.npz'.format(study_path, subj, subj, cond, ref, win)
    data = np.load(filename)

    con = data['con_tril'][:-1, :-1, :]
    con_mat = data['con_mat'][:-1, :-1, :]
    freqs = len(data['freqs'])
    n_ch = len(data['ch_names']) - 1

    con = con[np.tril_indices(n_ch, k=-1)]

    values = list()
    con_tril = [x for x in con.T]

    if win == '250ms':
        pos = [3, 4, 5, 6, 7, 8]
    else:
        pos = [1, 2, 3, 4, 5, 6, 7, 8]
    return con, con_mat, freqs, n_ch, con_tril, pos


def calc_electrode_dist(ch_info):
    pairs = list()
    distances = list()
    dist_mat = np.zeros((len(ch_info), len(ch_info)))
    for ix1, ch_1 in ch_info.iterrows():
        x1, y1, z1 = ch_1['natX'], ch_1['natY'], ch_1['natZ']
        name1 = ch_1['Electrode']
        for ix2 in range(ix1 + 1, len(ch_info)):
            ch_2 = ch_info.loc[ix2]
            x2, y2, z2 = ch_2['natX'], ch_2['natY'], ch_2['natZ']
            name2 = ch_2['Electrode']
            a = (x1, y1, z1)
            b = (x2, y2, z2)
            dist = distance.euclidean(a, b)
            distances.append(dist)
            pairs.append('{} | {}' .format(name1, name2))
            dist_mat[ix1, ix2] = round(dist, 2)
            # print('{} - {} mm' .format(pairs[-1], distances[-1]))
    return pairs, distances, dist_mat
