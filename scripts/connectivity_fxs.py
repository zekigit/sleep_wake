import numpy as np

def binarize(con, percentile):
    for fq in range(con.shape[2]):
        adj_mat = con[:, :, fq]
        values = adj_mat[np.tril_indices(adj_mat.shape[0])]
        per = np.percentile(values, percentile)
        adj_mat_bin = np.copy(adj_mat)
        adj_mat_bin[adj_mat_bin < per] = int(0)
        adj_mat_bin[adj_mat_bin >= per] = int(1)
        if fq == 0:
            binary_mats = adj_mat_bin
        else:
            binary_mats = np.dstack((binary_mats, adj_mat_bin))

    return binary_mats


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
    raw.add_events(events)
    return raw, events


def create_con_mat(con, freqs):
    con_mat = np.copy(con)
    upper = np.triu_indices_from(con_mat[:, :, 0])
    for fq in range(len(freqs)):
        swap = np.swapaxes(con_mat[:, :, fq], 0, 1)
        for val in zip(upper[0], upper[1]):
            con_mat[:, :, fq][val] = swap[val]
    return con_mat


def make_bnw_nodes(file_nodes, channels, colors, sizes):
    nodes = np.column_stack((channels, sizes, colors))
    np.savetxt(file_nodes, nodes, delimiter='\t')


def make_bnw_edges(file_nodes, adj_mats, titles):
    for idx, fq in enumerate(range(adj_mats.shape[2])):
        np.savetxt(file_nodes + titles[idx] + '.edge', adj_mats[:, :, idx], delimiter='\t')


def reshape_wsmi(columnas):
    rows = columnas.shape[0]
    y = 2 * rows
    y = y + (-1 / 2) ** 2
    y = np.sqrt(y)
    y = y*2+1
    y = y/2
    nb_chan = int(y)

    if len(columnas.shape) > 1:
        trials = columnas.shape[1]
    else:
        trials = 1

    mat_out = np.empty([nb_chan, nb_chan, trials])
    mat_out[:] = np.nan

    for tr in range(0, trials-1):
        n = 0
        for ch1 in range(0, nb_chan):
            for ch2 in range(ch1+1, nb_chan):
                mat_out[ch1, ch2, tr] = columnas[n, tr]
                mat_out[ch2, ch1, tr] = columnas[n, tr]
                n = n + 1

    return np.array(mat_out), nb_chan


import scipy.io as spio


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict



import neo
import numpy as np
import mne

# Example
# dirname = './data_Micromed/'
# trc_filename = 'EEG_33.TRC'
# fname = os.path.join(dirname, trc_filename)

def raw_from_neo(fname):
    seg_micromed = neo.MicromedIO(filename=fname).read_segment()

    ch_names = [sig.name for sig in seg_micromed.analogsignals]

    # Because here we have the same on all chan
    sfreq = seg_micromed.analogsignals[0].sampling_rate

    data = np.asarray(seg_micromed.analogsignals)
    data *= 1e-6  # putdata from microvolts to volts
    # add stim channel
    ch_names.append('STI 014')
    data = np.vstack((data, np.zeros((1, data.shape[1]))))

    # To get sample number:
    events_time = seg_micromed.eventarrays[0].times.magnitude * sfreq
    n_events = len(events_time)
    events = np.empty([n_events, 3])
    events[:, 0] = events_time
    events[:, 2] = seg_micromed.eventarrays[0].labels.astype(int)

    ch_types = ['eog', 'eog'] + ['eeg' for _ in ch_names[2:-1]] + ['stim']
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)
    raw.add_events(events)
    return raw
