import numpy as np


def mdict_to_matrix(M_dict, source_labels, target_labels):
    Mtot = np.zeros((len(source_labels), len(target_labels)))
    for l, M in M_dict.items():
        Mtot[
            np.ix_(np.where(source_labels == l)[0], np.where(target_labels == l)[0])
        ] = M
    return Mtot
