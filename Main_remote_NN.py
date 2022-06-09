import argparse
import timeit
import numpy as np

import Config_exp
import Config_plot
from classConfig import Config
from sub_NNdata import NNdata
from sub_NNrescaling import NNrescaling
from Datas.classSignal import SignalForce, SignalImg
from Utils.classPlot import ClassPlot
from Utils.classStat import Histo
from sub_scalarevent import scalarevent
from sub_scalarstats import scalarstats

################### Parser ##################################

parser = argparse.ArgumentParser(description="run data analyse")
parser.add_argument("-r", "--remote", action="store_true", help="n'affiche pas les plots")
parser.add_argument("p", type=str, help="path_from_root")
parser.add_argument("t", type=str, help="référence du tricot")
parser.add_argument("n", type=str, help="référence de l'expérience")
parser.add_argument("v", type=str, help="version sous laquelle on travaille")
parser.add_argument("--fluNN", action="store_true", help="create NN data")
parser.add_argument("--flurscNN", action="store_true", help="rsc NN data scalaire")
parser.add_argument("--scalarstats", action="store_true", help="run partie scalarstats")
parser.add_argument("--scalarevent", action="store_true", help="run partie scalarevent")

args = parser.parse_args()

# %% ################### arg Set up ##################################

remote = args.remote

ref_tricot = args.t
n_exp = args.n
version_work = args.v

NAME_EXP = ref_tricot + n_exp + version_work

print(NAME_EXP)


# %% ################### direct Set up ##################################

display_figure = True
display_figure_reg = False

# %% ################### main code##################################

config = Config(args.path_from_root, Config_exp.exp[NAME_EXP])
config_plot = Config_plot.plot[NAME_EXP]
histo = Histo(config)
plot = ClassPlot(remote, histo)

signaltype = 'flu'
NN_data = ''
signal_flu = SignalForce(config, signaltype, NN_data)
if config.img:
    signal_img = SignalImg(config, signaltype, NN_data, fields=False)
else:
    signal_img = None

if args.fluNN:
    start_time = timeit.default_timer()
    print('______ on va creer des NN_data à partir de flu ______')

    NNdata(config, remote, signaltype, signal_flu, signal_img)

    stop_time = timeit.default_timer()
    print('______ tps pour creer les NN data : {} ______'.format(stop_time - start_time))

if args.flurscNN:
    start_time = timeit.default_timer()
    print('______ on va rsc les flu NN_data ______')

    NN_data = ['train', 'val', 'test']
    signal_flu = [SignalForce(config, signaltype, NN_data[i]) for i in range(np.size(NN_data))]
    if config.img:
        signal_img = [SignalImg(config, signaltype, NN_data[i]) for i in range(np.size(NN_data))]
    else:
        signal_img = [None for i in range(np.size(NN_data))]

    NNrescaling(config, plot, signaltype, NN_data, signal_flu, signal_img, display_figure)

    stop_time = timeit.default_timer()
    print('______ tps pour rsc les flu NN data : {} ______'.format(stop_time - start_time))

NN_data = 'train'
signaltype = 'flu_rsc'
signal_flu = SignalForce(config, signaltype, NN_data)

if args.scalarstats:
    start_time = timeit.default_timer()
    print('______ on entre dans scalarstats ______')
    scalarstats(config, signaltype, NN_data, plot, signal_flu)
    stop_time = timeit.default_timer()
    print('______ tps pour scalarstats : {} ______'.format(stop_time - start_time))

signal_img = None
if config.img:
    signal_img = SignalImg(config, signaltype, NN_data, fields=False)
else:
    signal_img = None

if args.scalarevent:
    start_time = timeit.default_timer()
    print('______ on entre dans scalarevent ______')
    scalarevent(config, config_plot, signaltype, NN_data, histo, plot, display_figure, display_figure_reg,
                    signal_flu, signal_img, Sm=False)

    stop_time = timeit.default_timer()
    print('______ tps pour scalarevent : {} ______'.format(stop_time - start_time))