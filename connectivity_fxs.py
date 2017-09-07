import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mne
from scipy.fftpack import rfft, irfft
from scipy import stats
import pandas as pd


def binarize(con_mat, percentile=None, sd=None, value=None):
    values = con_mat[np.tril_indices(con_mat.shape[0], k=-1)]
    if percentile:
        val = np.percentile(values, percentile)
    elif sd:
        val = np.median(values) + sd*np.std(values)
    elif value:
        val = value
    else:
        print('You should specify a threshold')
        return

    adj_mat_bin = np.copy(con_mat)
    adj_mat_bin[adj_mat_bin < val] = int(0)
    adj_mat_bin[adj_mat_bin >= val] = int(1)
    return adj_mat_bin


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


def make_bnw_nodes(file_nodes, coords, colors, sizes):
    if isinstance(colors, float):
        colors = [colors] * len(coords)
    if isinstance(sizes, float):
        sizes = [sizes] * len(coords)

    nodes = np.column_stack((coords, colors, sizes))
    np.savetxt(file_nodes, nodes, delimiter='\t')


def make_bnw_edges(file_nodes, adj_mats, titles):
    for idx, fq in enumerate(range(adj_mats.shape[2])):
        np.savetxt(file_nodes + titles[idx] + '.edge', adj_mats[:, :, idx], delimiter='\t')


def find_threshold(mat):
    values = mat[np.tril_indices(mat.shape[0], k=-1)]
    median = np.median(values)
    sd = np.std(values)
    threshold = median + sd
    hist = plt.hist(values)
    plt.vlines(threshold, ymin=0, ymax=max(hist[0]))
    return threshold


def binarize_mat(mat, threshold):
    bin_mat = mat.copy()
    bin_mat[bin_mat > threshold] = 1
    bin_mat[bin_mat < threshold] = 0

    return bin_mat


def graph_threshold(mat, steps):
    avgs = np.empty((len(steps)))
    stds = np.empty((len(steps)))
    vals = list()

    for ix, s in enumerate(steps):
        copy_mat = mat.copy()
        copy_mat[copy_mat > s] = 1
        copy_mat[copy_mat < s] = 0
        graph = nx.from_numpy_matrix(copy_mat)
        degs = np.array(list(graph.degree().values()))/len(graph)
        avgs[ix] = np.average(degs)
        stds[ix] = np.std(degs)
        vals.extend(degs)
    df_deg = {'Degree': vals}
    deg_df = pd.DataFrame(df_deg)
    # plt.plot(steps, avgs)
    return avgs, stds, deg_df


def calc_graph_metrics(bin_mat, ch_names):
    graph = nx.from_numpy_matrix(bin_mat)
    for ix, n in enumerate(graph):
        graph.node[ix] = ch_names[ix]

    degree = graph.degree()
    # nx.draw_networkx(graph, with_labels=True)
    # nx.draw(graph)
    clustering = nx.clustering(graph)
    # avg_path_length = nx.average_shortest_path_length(graph)
    return clustering, degree, graph


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
        mat_out = np.empty([nb_chan, nb_chan, trials])
        mat_out[:] = np.nan

        for tr in range(trials):
            n = 0
            for ch1 in range(0, nb_chan):
                for ch2 in range(ch1 + 1, nb_chan):
                    mat_out[ch1, ch2, tr] = columnas[n, tr]
                    mat_out[ch2, ch1, tr] = columnas[n, tr]
                    n += 1

    else:
        trials = 1

        mat_out = np.empty([nb_chan, nb_chan])
        mat_out[:] = np.nan

        for tr in range(trials):
            n = 0
            for ch1 in range(0, nb_chan):
                for ch2 in range(ch1+1, nb_chan):
                    mat_out[ch1, ch2] = columnas[n]
                    mat_out[ch2, ch1] = columnas[n]
                    n += 1

    return np.array(mat_out), nb_chan


def phase_scramble_ts(ts):
    """Returns a TS: original TS power is preserved; TS phase is shuffled."""
    fs = rfft(ts)
    # rfft returns real and imaginary components in adjacent elements of a real array
    pow_fs = fs[1:-1:2]**2 + fs[2::2]**2
    phase_fs = np.arctan2(fs[2::2], fs[1:-1:2])
    phase_fsr = phase_fs.copy()
    np.random.shuffle(phase_fsr)
    # use broadcasting and ravel to interleave the real and imaginary components.
    # The first and last elements in the fourier array don't have any phase information, and thus don't change
    fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr), np.sin(phase_fsr)]
    fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
    tsr = irfft(fsrp)
    return tsr


def randomize_epochs_phase(epochs):
    dat = epochs.get_data()
    if dat.shape[2] & 0x1:
        dat = dat[:, :, :-1]

    shuf_dat = np.empty(dat.shape)
    for ix_e, e in enumerate(dat):
        for ix_ch, ch in enumerate(e):
            shuf_signal = phase_scramble_ts(ch)
            shuf_dat[ix_e, ix_ch, :] = shuf_signal
    shuf_epo = mne.EpochsArray(shuf_dat, epochs.info)
    return shuf_epo


def d3_scale(dat, out_range=(-1, 1)):
    domain = [np.min(dat, axis=0), np.max(dat, axis=0)]

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b = 1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat))


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



