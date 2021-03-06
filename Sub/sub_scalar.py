# %%
import numpy as np
import scipy.io as spo
import pandas as pd
from pathlib import Path

from Datas.classScalar import Preprocess, Flu


def scalar(config, plot, remote):

    prepro = Preprocess(config, remote)
    fluctuations = Flu(config, remote)


#  %% ################### Partie 1 : Preprocess ##################################

    if not config.mix_set:

        toload_rawdata = config.global_path_load_raw + 'parse_data/SpecimenRawData1_matrix'

        data = spo.loadmat(toload_rawdata)
        data = data['SpecimenRawData1_matrix']

        if config.img:
            toload_picture_time = config.global_path_load_raw + 'parse_data/picture_time.csv'
            toload_trigger_time = config.global_path_load_raw + 'parse_data/trigger_time.csv'

            picturetimeraw = pd.read_csv(toload_picture_time, sep=',', header=None)
            triggertimeraw = pd.read_csv(toload_trigger_time, sep=',', header=None)

            picturetimeraw = picturetimeraw.values
            triggertimeraw = triggertimeraw.values
        else:
            picturetimeraw = None
            triggertimeraw = None

        indice_lw1, Lw_force_ref, f_ref, m_f, m_trenorm, m_ext, m_size, index_picture_m, number_picture_m = prepro.main_run(
            plot,
            data,
            picturetimeraw,
            triggertimeraw)

    else:

        m_size = np.array([0])
        indice_lw1 = np.array([0])
        Lw_f_ref = np.array([0])
        f_ref = np.array([0])

        m_f = []
        m_trenorm = []
        m_ext = []

        index_picture_m = []
        number_picture_m = []

        for i in range(config.nb_set):
            toload_rawdata = config.global_path_load_raw % (config.date[i], config.nexp[i], config.version_raw) + 'm_f/'

            indice_lw1_sub, Lw_f_ref_sub, f_ref_sub, m_f_sub, m_trenorm_sub, m_ext_sub, m_size_sub, index_picture_m_sub, \
            number_picture_m_sub = prepro.import_mix(toload_rawdata, num_set=i)

            if config.img:
                index_picture_m = index_picture_m + index_picture_m_sub
                number_picture_m = number_picture_m + number_picture_m_sub
            else:
                index_picture_m = None
                number_picture_m = None

            m_size = np.hstack((m_size, m_size_sub[1::]))
            indice_lw1 = np.hstack((indice_lw1, indice_lw1_sub[1::]))
            Lw_f_ref = np.hstack((Lw_f_ref, Lw_f_ref_sub[1::]))
            f_ref = np.hstack((f_ref, f_ref_sub[1::]))
            m_f = m_f + m_f_sub
            m_trenorm = m_trenorm + m_trenorm_sub
            m_ext = m_ext + m_ext_sub

        indice_lw1 = indice_lw1[1::]
        Lw_f_ref = Lw_f_ref[1::]
        f_ref = f_ref[1::]
        m_size = m_size[1::]
        f_min_size = np.min(m_size)

        # Define the colors to be used using rainbow map (or any other map)
        colors = plot.make_colors(np.size(Lw_f_ref))

        # Define the grig quand plusieurs set
        grid = plot.make_grid(config)

        plot.plot_x_y_multiliste(np.size(Lw_f_ref), Lw_f_ref, f_ref, 'L_{w} (mm)', 'F(N)', 'Lw_drift', 'F',
                                save=prepro.to_save_fig, title='caracterisation  drift : force ?? extension de changement de regime',
                                colors=colors)

        plot.plot_y(Lw_f_ref, 'cycle', 'L_{w} (mm)', 'cycles', 'Lw_drift',
                   save=prepro.to_save_fig,
                   title='caracterisation  drift : extension de changement de regime',
                   grid=grid)

    # %% ################## Partie 2 : Flu ##################################

    fileName = fluctuations.tosave_beta + '_1' + '.npy'
    fileObj = Path(fileName)
    fit_done = fileObj.is_file()
    print(fluctuations.tosave_beta)
    beta, flu, t_flu, ext_flu, flu_size, Lw_0_flu, index_picture_flu, number_picture_flu = fluctuations.main_run(plot,
                                                                                                                 m_f,
                                                                                                                 m_ext,
                                                                                                                 m_trenorm,
                                                                                                                 index_picture_m,
                                                                                                                 number_picture_m,
                                                                                                                 fit_done=fit_done)