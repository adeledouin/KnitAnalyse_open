import argparse
import timeit
import numpy as np

import Config_exp
import Config_plot
from classConfig import Config
from sub_scalar import scalar
from sub_scalarevent import scalarevent
from Datas.classSignal import SignalForce, SignalImg
from Datas.classEvent import InfoField
from Utils.classStat import Histo
from Utils.classPlot import ClassPlot
from sub_transfert_csv import transfert_csv
from sub_scalarstats import scalarstats

################### Parser ##################################

parser = argparse.ArgumentParser(description="run data analyse")
parser.add_argument("-r", "--remote", action="store_true", help="n'affiche pas les plots")
parser.add_argument("p", type=str, help="path_from_root")
parser.add_argument("t", type=str, help="référence du tricot")
parser.add_argument("n", type=str, help="référence de l'expérience")
parser.add_argument("v", type=str, help="version sous laquelle on travaille")
parser.add_argument("--scalar", action="store_true", help="run partie scalar")
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

signaltype = 'flu'
NN_data = ''
ftype = 'img'
add_sub = ''


# %% ################### main code##################################

config = Config(args.path_from_root, Config_exp.exp[NAME_EXP])
config_plot = Config_plot.plot[NAME_EXP]
histo = Histo(config)
plot = ClassPlot(remote, histo)

if args.scalar:
    start_time = timeit.default_timer()

    print('______ on entre dans scalar ______')
    scalar(config, plot, remote)

    stop_time = timeit.default_timer()
    print('______ tps pour scalar : {} ______'.format(stop_time - start_time))


start_time = timeit.default_timer()
print('______ on charge signal scalar ______')
signal_flu = SignalForce(config, signaltype, NN_data)
transfert_csv(config, signal_flu)

if args.scalarstats:
    start_time = timeit.default_timer()
    print('______ on entre dans scalarstats ______')
    scalarstats(config, signaltype, NN_data, plot, signal_flu)
    stop_time = timeit.default_timer()
    print('______ tps pour scalarstats : {} ______'.format(stop_time - start_time))

signal_img = None

if args.scalarevent:
    start_time = timeit.default_timer()
    print('______ on entre dans scalarevent ______')
    scalarevent(config, config_plot, signaltype, NN_data, histo, plot, display_figure, display_figure_reg,
                    signal_flu, signal_img, Sm=True)

    scalarevent(config, config_plot, signaltype, NN_data, histo, plot, display_figure, display_figure_reg,
                    signal_flu, signal_img, Sm=False)

    stop_time = timeit.default_timer()
    print('______ tps pour scalarevent : {} ______'.format(stop_time - start_time))

