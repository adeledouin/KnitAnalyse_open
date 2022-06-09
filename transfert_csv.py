# %%
import numpy as np

from classConfig import Config
from Utils.classCell import Cell
from Datas.classSignal import SignalForce
import Config_exp

################### Variables ##################################
remote = False

ref_tricot = 'knit12_'
n_exp = '201029_exp1_'
version_work = 'v1'
path_from_root = '/path_from_root/'
NAME_EXP = ref_tricot + n_exp + version_work

config = Config(path_from_root, Config_exp.exp[NAME_EXP])

signaltype = 'flu'
NN_data = ''
display_figure = True
display_figure_reg = False

################### Main code ##################################

signal = SignalForce(config, signaltype, '')

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
