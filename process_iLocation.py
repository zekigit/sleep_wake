import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wake_sleep_info import study_path
from ieeg_fx import read_ch_info

pd.set_option('display.expand_frame_repr', False)


subj = 's5'
markups = pd.read_excel('{}/{}/info/{}_slicer_locs.xlsx' .format(study_path, subj, subj))
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

    for c in range(n-1):
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
        chan = '{}{}' .format(s, ix+1)
        if chan in ch_info['Electrode'].values:
            ch_info.set_value(ch_info['Electrode'] == chan, 'natX', c[0])
            ch_info.set_value(ch_info['Electrode'] == chan, 'natY', c[1])
            ch_info.set_value(ch_info['Electrode'] == chan, 'natZ', c[2])
