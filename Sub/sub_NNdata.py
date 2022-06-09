# %%
import numpy as np

from Datas.classSignal import SignalForce
from Datas.classNNData import CreateNNData

def NNdata(config, remote, signaltype, signal_flu, signal_img):

    create_NN_data = CreateNNData(config, signaltype, set_to_keep=None) #np.array([3, 4, 5, 6, 7, 8, 9]).astype(int))

    NN_data = ['train', 'val', 'test']
    NN_data_cycles = [create_NN_data.train_cycle, create_NN_data.val_cycle, create_NN_data.test_cycle]
    if config.mix_set:
        sub_cycles_NN_data = [create_NN_data.train_sub_cycle_NN, create_NN_data.val_sub_cycle_NN,
                              create_NN_data.test_sub_cycle_NN]
        NN_data_sub_cycles = [create_NN_data.train_NN_sub_cycle, create_NN_data.val_NN_sub_cycle,
                              create_NN_data.test_NN_sub_cycle]

    for i in range(np.size(NN_data)):
        NN = NN_data[i]
        cycles = NN_data_cycles[i]

# %% ################### Partie 1 : Separe flu data ##################################
        if not config.mix_set:
            NN_sub_cycles = None
            sub_cycles_NN = None
            f_NN, t_NN, ext_NN, index_picture_NN, number_picture_NN, numbertot_picture_NN, nb_index_picture_NN = create_NN_data.create_NN_data(
                signal_flu.f, signal_flu.ext, signal_flu.t, signal_img.index_picture,
                signal_img.number_picture, cycles)
        else:
            sub_cycles_NN = sub_cycles_NN_data[i]
            NN_sub_cycles = NN_data_sub_cycles[i]
            f_NN, t_NN, ext_NN, index_picture_NN, number_picture_NN, numbertot_picture_NN, nb_index_picture_NN = create_NN_data.create_NN_data(
                signal_flu.f, signal_flu.ext, signal_flu.t, signal_img.index_picture if signal_img is not None else None,
                signal_img.number_picture if signal_img is not None else None, cycles,
                signal_img.numbertot_picture if signal_img is not None else None, sub_cycles_NN)

        signal_flu.save_data(signaltype, NN, f_NN, t_NN, ext_NN, np.size(f_NN[0, :]),
                             index_picture_NN, number_picture_NN, numbertot_picture_NN, nb_index_picture_NN,
                             np.size(cycles), cycles, None, sub_cycles_NN, NN_sub_cycles)

        signal_NN = SignalForce(config, signaltype, NN)

# %% ################### Partie 2 : Separe img data ##################################

        if config.img:
            if not config.mix_set:
                prev_field = [signal_img.vit_x, signal_img.vit_y, signal_img.vit_X, signal_img.vit_Y, signal_img.slip_x,
                              signal_img.slip_y, signal_img.slip_X, signal_img.slip_Y, signal_img.vort, signal_img.dev,
                              signal_img.shear, signal_img.div, signal_img.posx, signal_img.posy, signal_img.posX,
                              signal_img.posY]
            else:
                prev_field = None

            names_tosave = ['vit_x', 'vit_y', 'vit_x_XY', 'vit_y_XY', 'slip_x', 'slip_y', 'slip_x_XY', 'slip_y_XY',
                            'vort',
                            'dev', 'shear', 'div', 'posx', 'posy', 'posx_XY', 'posy_XY']

            signal_img.reshape_all_fields(names_tosave, signal_NN, prev_field, sub_NN=True)

            signal_img.save_data(signaltype, NN, None, None, None, None, None, None, None, None,
                                 np.size(cycles), cycles, None, sub_cycles_NN, NN_sub_cycles)


