# %%
import numpy as np

from classConfig import Config
from Datas.classSignal import SignalForce, SignalImg, VariationsScalar
from Utils.classPlot import ClassPlot
from Utils.classStat import Histo
from Sub.sub_plot_variations import plot_variations_flu
import Config_exp
import Config_plot



################### Main code ##################################

remote = False

ref_tricot = 'knit005_'
n_exp = 'mix_'
version_work = 'v1'
path_from_root = '/path_from_root/'
NAME_EXP = ref_tricot + n_exp + version_work

print(NAME_EXP)

config = Config(path_from_root, Config_exp.exp[NAME_EXP])
config_plot = Config_plot.plot[NAME_EXP]

histo = Histo(config)
plot = ClassPlot(remote, histo)

display_figure = True
display_figure_variations = False
display_figure_reg = False

# %% ################### Partie 0 : Recup flu data ##################################

NN_data = ['train', 'val', 'test']

signaltype = 'flu'
signal_flu = [SignalForce(config, signaltype, NN_data[i]) for i in range(np.size(NN_data))]
if config.img:
    signal_img = [SignalImg(config, signaltype, NN_data[i]) for i in range(np.size(NN_data))]
else:
    signal_img = [None for i in range(np.size(NN_data))]

if display_figure:
    plot.plot_x_y_multiarray(signal_flu[0].nbcycle, signal_flu[0].ext, signal_flu[0].f,
                             'L_{w} (mm)', '\delta f', 'Lw', 'f', pts='-')


# %% ################### Partie 1 : Rsc flu ##################################
if not config.mix_set:
    variations_flu = VariationsScalar(config, pourcentage=5, f=signal_flu[0].f, ext=signal_flu[0].ext,
                                      t=signal_flu[0].t,
                                      index=None, number=None, directsignal=True,
                                      signaltype=signaltype, NN_data=NN_data[0], ftype='force', fname=signaltype,
                                      rsc=True, stats=False) ## à mettre True plus tard

    for i in range(np.size(NN_data)):
        NN = NN_data[i]

        f_rsc, ext_rsc, t_rsc, f_rsc_size, index_picture_rsc, number_picture_rsc, _,\
        nb_index_picture_rsc = variations_flu.rsc_par_fenetre_tps(signal_flu[i].f, signal_flu[i].ext, signal_flu[i].t,
                                                                  signal_img[i].index_picture,
                                                                  signal_img[i].number_picture)

        signal_flu[i].save_data('flu_rsc', NN, f_rsc, t_rsc, ext_rsc, f_rsc_size, index_picture_rsc, number_picture_rsc,
                                None, nb_index_picture_rsc, signal_flu[i].nbcycle, signal_flu[i].cycles,
                                signal_flu[i].sub_cycles, signal_flu[i].sub_cycles_NN, signal_flu[i].NN_sub_cycles)

else:
    variations_mix = [0 for i in range(config.nb_set)]
    for i in range(config.nb_set):
        variations_mix[i] = VariationsScalar(config, pourcentage=5, f=signal_flu[0].f[signal_flu[0].NN_sub_cycles[i], :],
                                             ext=signal_flu[0].ext[signal_flu[0].NN_sub_cycles[i], :],
                                             t=signal_flu[0].t[signal_flu[0].NN_sub_cycles[i], :],
                                             index=None, number=None, directsignal=True,
                                             signaltype=signaltype, NN_data=NN_data[0], ftype='force', fname=signaltype,
                                             rsc=True)

    ## rsc flu
    for j in range(np.size(NN_data)):
        NN = NN_data[j]

        size_rsc = variations_mix[0].size_rsc_array(signal_flu[j].t[0, :])

        f_rsc = np.zeros((signal_flu[j].nbcycle, size_rsc))
        ext_rsc = np.zeros((signal_flu[j].nbcycle, size_rsc))
        t_rsc = np.zeros((signal_flu[j].nbcycle, size_rsc))
        index_picture_rsc = np.zeros((signal_flu[j].nbcycle, size_rsc))
        number_picture_rsc = np.zeros((signal_flu[j].nbcycle, size_rsc))
        numbertot_picture_rsc = np.zeros((signal_flu[j].nbcycle, size_rsc))
        nb_index_picture_rsc = np.zeros(config.nb_set)

        for i in range(config.nb_set):
            if signal_flu[j].NN_sub_cycles[i].size == 0 and NN == 'train':
                print('issues : no train data for set {}'.format(i))
            elif signal_flu[j].NN_sub_cycles[i].size != 0:
                    f_rsc[signal_flu[j].NN_sub_cycles[i], :], ext_rsc[signal_flu[j].NN_sub_cycles[i], :], t_rsc[
                                                                                                          signal_flu[
                                                                                                              j].NN_sub_cycles[
                                                                                                              i], :], \
                    f_rsc_size, index_picture_rsc[signal_flu[j].NN_sub_cycles[i], :], \
                    number_picture_rsc[signal_flu[j].NN_sub_cycles[i], :], numbertot_picture_rsc[
                                                                           signal_flu[j].NN_sub_cycles[i], :], \
                    nb_index_picture_rsc[i] = variations_mix[i].rsc_par_fenetre_tps(
                        signal_flu[j].f[signal_flu[j].NN_sub_cycles[i], :],
                        signal_flu[j].ext[signal_flu[j].NN_sub_cycles[i], :],
                        signal_flu[j].t[signal_flu[j].NN_sub_cycles[i], :],
                        signal_img[j].index_picture[
                        signal_flu[j].NN_sub_cycles[i], :] if
                        signal_img[
                            j] is not None else None,
                        signal_img[j].number_picture[
                        signal_flu[j].NN_sub_cycles[i], :] if
                        signal_img[
                            j] is not None else None,
                        signal_img[j].numbertot_picture[
                        signal_flu[j].NN_sub_cycles[i], :] if
                        signal_img[
                            j] is not None else None,
                        NN)

        signal_flu[j].save_data('flu_rsc', NN, f_rsc, t_rsc, ext_rsc, size_rsc, index_picture_rsc, number_picture_rsc,
                                numbertot_picture_rsc, nb_index_picture_rsc, signal_flu[j].nbcycle, signal_flu[j].cycles,
                                signal_flu[j].sub_cycles, signal_flu[j].sub_cycles_NN, signal_flu[j].NN_sub_cycles)

signaltype = 'flu_rsc'
signal_flu_rsc = [SignalForce(config, signaltype, NN_data[i]) for i in range(np.size(NN_data))]

if not config.mix_set:
    for j in range(np.size(NN_data)):
        NN = NN_data[j]

        if display_figure:
            plot.plot_x_y_multiarray(signal_flu_rsc[j].nbcycle, signal_flu_rsc[j].ext, signal_flu_rsc[j].f,
                                     'L_{w} (mm)', '\delta f', 'Lw', signal_flu_rsc[j].fname, save=signal_flu_rsc[j].to_save_fig)

        if display_figure_variations:
            variations_flu_rsc = VariationsScalar(config, pourcentage=5, f=signal_flu_rsc[j].f,
                                                  ext=signal_flu_rsc[j].ext,
                                                  t=signal_flu_rsc[j].t,
                                                  index=None, number=None, directsignal=True,
                                                  signaltype=signaltype, NN_data=NN, ftype='force', fname=signaltype)

            plot_variations_flu(signal_flu_rsc[j], variations_flu_rsc)

# %% ################### Partie 2 : ReshapeImg ##################################

if config.img:
    for i in range(np.size(NN_data)):
        NN = NN_data[i]

        if not config.mix_set:
            prev_field = [signal_img[i].vit_x, signal_img[i].vit_y, signal_img[i].vit_X, signal_img[i].vit_Y,
                          signal_img[i].slip_x,
                          signal_img[i].slip_y, signal_img[i].slip_X, signal_img[i].slip_Y, signal_img[i].vort,
                          signal_img[i].dev,
                          signal_img[i].shear, signal_img[i].div, signal_img[i].posx, signal_img[i].posy,
                          signal_img[i].posX, signal_img[i].posY]
        else:
            prev_field = None

        names_tosave = ['vit_x', 'vit_y', 'vit_x_XY', 'vit_y_XY', 'slip_x', 'slip_y', 'slip_x_XY', 'slip_y_XY', 'vort',
                        'dev', 'shear', 'div', 'posx', 'posy', 'posx_XY', 'posy_XY']

        signal_img[i].reshape_all_fields(names_tosave, signal_flu_rsc[i], None, sub_NN=True)

        signal_img[i].save_data(signaltype, NN, None, None, None, None, None, None, None, None,
                             np.size(signal_flu_rsc[i].cycles), signal_flu_rsc[i].cycles, None,
                             signal_flu_rsc[i].sub_cycles_NN, signal_flu_rsc[i].NN_sub_cycles)


