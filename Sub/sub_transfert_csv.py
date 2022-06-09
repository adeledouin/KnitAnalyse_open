# %%
import numpy as np
from Utils.classCell import Cell


################### Main code ##################################

def transfert_csv(config, signal):

    if not config.mix_set:
        maxcycle = config.maxcycle
        nbcycle = config.nbcycle
        mincycle = config.mincycle
    else:
        maxcycle = np.sum(config.maxcycle)
        nbcycle = np.sum(config.nbcycle)
        mincycle = config.mincycle[0]

    Cell(signal.path_signal + 'f', 1, data=signal.f, extension='csv')
    Cell(signal.path_signal + 'ext', 1, data=signal.ext, extension='csv')
    Cell(signal.path_signal + 't', 1, data=signal.t, extension='csv')