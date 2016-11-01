# Criticality Functions
# Ezequiel Mikulan - emikulan@gmail.com

import numpy as np

def avalanchas(datos):
    bin_mat = np.empty([0, datos.shape[1]])
    print('Calculating avalanches,  Channels x Samples:', bin_mat.shape)

    for ch in datos:
        ch_std = np.std(ch)*3.5
        ch_bin = np.where(abs(ch) > ch_std, 1, 0)
        bin_mat = np.vstack((bin_mat, ch_bin))


    av_size = np.empty([0,0])
    id1 = 0

    for ind, col in enumerate(bin_mat.T):
        suma = sum(col)
        if suma == 0:
            id2 = ind
            av_aux = bin_mat[:, id1:id2]

            if av_aux.shape[1]>1:
                size_aux = np.sum(av_aux)
                av_size = np.append(av_size, size_aux)

            id1 = id2
    return bin_mat, av_size


def sort_and_prob(av_size):
    sorted_av_size = np.sort(av_size)[::-1]
    n_avalanche = len(sorted_av_size)
    prob = np.array(range(n_avalanche))/n_avalanche
    return sorted_av_size, prob

