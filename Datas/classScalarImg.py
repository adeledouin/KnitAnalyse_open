import timeit
import scipy.io as spo
from pathlib import Path
import numpy as np

from Utils.classCell import Cell
from Utils.classStat import Stat, Shape


# ---------------------------------------------------------------------------------------------------------------------#
class ReshapeImgFlu():
    '''  '''

    # ---------------------------------------------------------#
    def __init__(self, config, remote, display_figure=False, saving_step=True,
                 way_to_cut_img='shrink'):
        ''' display_figure : affiche les figures pour vérifier pas de pb dans l'analyse
            saving step : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas'''

        ## Config
        self.config = config
        self.remote = remote

        self.nb_process = config.nb_process
        self.display_figure = display_figure
        self.saving_step = saving_step
        self.way_to_cut_img = way_to_cut_img

        if not self.config.mix_set:
            self.nbcycle = self.config.nbcycle
        else:
            self.nbcycle = np.sum(self.config.nbcycle)
            self.sub_cycles = self.config.sub_cycles

        if not self.config.mix_set:
            self.to_load = self.config.global_path_load + 'pict/'
            self.to_save = self.config.global_path_save + '/pict_event_flu/'

            self.toload_index_picture = self.config.global_path_save + 'flu/index_picture_flu' + '.npy'
            self.toload_number_picture = self.config.global_path_save + 'flu/number_picture_flu' + '.npy'
            self.toload_nb_index_picture = self.config.global_path_save + 'flu/nb_index_picture_flu' + '.npy'

            self.index_picture = np.load(self.toload_index_picture)
            self.number_picture = np.load(self.toload_number_picture)
            self.nb_index_picture = np.load(self.toload_nb_index_picture)

        else:
            self.to_save = self.config.global_path_save + '/pict_event_flu/'

    # ------------------------------------------
    def mat_to_py(self, path, name):

        data = spo.loadmat(path)
        data = data[name]

        return data

    # ------------------------------------------
    def import_data(self, to_load, name_toload, name_matvariable):
        if not self.config.mix_set:
            start_time = timeit.default_timer()

        toload_field = to_load + name_toload
        field = self.mat_to_py(toload_field, name_matvariable)

        if not self.config.mix_set:
            stop_time = timeit.default_timer()
            print('tps pour importer :', stop_time - start_time)

        return field

    # ------------------------------------------
    def cut(self, array, min_lw, min_lc, max_lw, max_lc):

        shape = Shape(array)
        shape_lw = shape.size_w
        shape_lc = shape.size_c
        shape_picture = shape.nb_pict

        if self.way_to_cut_img == 'shrink':
            new_array = np.zeros((min_lw, min_lc, np.sum(shape_picture)))
            k = 0
            init_lc = int(np.round((shape_lc - min_lc) / 2))
            f_lc = int(init_lc + min_lc)
            init_lw = int(np.round((shape_lw - min_lw) / 2))
            f_lw = int(init_lw + min_lw)

            # print('min lw', min_lw, 'min_lc', min_lc)
            # print('new init lw', init_lw, 'new f lw', f_lw, 'new init lc', init_lc, 'new f lc', f_lc)

            new_array = array[init_lw:f_lw, init_lc:f_lc, :]

        elif self.way_to_cut_img == 'expand':
            new_array = np.zeros((max_lw, max_lc, shape_picture))
            init_lc = np.round((max_lc - shape_lc) / 2)
            init_lw = np.round((max_lw - shape_lw) / 2)

            # print('new init lc', init_lc, 'new init lw', init_lw)

        return new_array

    # ------------------------------------------
    def recup_shape(self, name_tosave):
        to_load_field = self.config.global_path_save + '/pict_event_flu/' + name_tosave + '.npy'

        field = np.load(to_load_field)
        shape = Shape(field)

        ##save
        tosave_shape_field = self.to_save + name_tosave + '_shape' + '.npy'
        np.save(tosave_shape_field, np.array([shape.size_w, shape.size_c, shape.nb_pict]))

    # ------------------------------------------
    def reshape_all_fields(self, names_toload, names_matvariables, names_tosave):
        for i in range(np.size(names_toload)):

            if not self.config.mix_set:
                fileName = self.to_save + names_tosave[i] + '.npy'
            else:
                fileName = self.to_save + names_tosave[i] + '_{}'.format(self.config.nb_set - 1) + '.npy'
            fileObj = Path(fileName)
            is_fileObj = fileObj.is_file()

            if not is_fileObj:
                print('reshape de {} has to be done'.format((names_tosave[i])))
                self.reshape_field(names_toload[i], names_matvariables[i], names_tosave[i])
            else:
                if not self.config.mix_set:
                    shapeName = self.to_save + names_tosave[i] + '_shape' + '.npy'
                    shapeObj = Path(shapeName)
                    is_shapeObj = shapeObj.is_file()

                    if not is_shapeObj:
                        print('recup shape de {} has to be done'.format((names_tosave[i])))
                        self.recup_shape(names_tosave[i])

                print('reshape de {} is alredy done'.format((names_tosave[i])))

    # ------------------------------------------
    def reshape_field(self, name_toload, name_matvariable, name_tosave):

        if not self.config.mix_set:

            # print('on cherche {} pict'.format(self.nb_index_picture))
            field = self.import_data(self.to_load, name_toload, name_matvariable)
            print('verif reshape img dans set : befor {} '.format(np.shape(field)))
            field, number_picture, _, nb_pict_set = self.reshape(np.arange(self.nbcycle),
                                                                 np.ones_like(self.number_picture) * np.nan,
                                                                 field, self.index_picture, self.number_picture)

            shape = Shape(field)
            print('verif reshape img dans set : nb_index_picture - nb_picture_set = {}'.format(
                self.nb_index_picture - nb_pict_set))
            print('verif reshape img dans set : after {}'.format(np.shape(field)))

            if self.saving_step:
                tosave_field = self.to_save + name_tosave + '.npy'
                tosave_shape_field = self.to_save + name_tosave + '_shape' + '.npy'

                np.save(tosave_field, field)
                np.save(tosave_shape_field, np.array([shape.size_w, shape.size_c, shape.nb_pict]))
        else:
            self.index_picture, number_picture, \
            numbertot_picture, self.nb_index_picture = self.reshape_mix(name_toload, name_matvariable, name_tosave)

            if self.saving_step:
                tosave_numbertot_picture = self.to_save + 'numbertot_picture_flu'
                np.save(tosave_numbertot_picture, numbertot_picture)

        if self.saving_step:
            tosave_index_picture = self.to_save + 'index_picture_flu'
            tosave_number_picture = self.to_save + 'number_picture_flu'
            tosave_new_number_picture = self.config.global_path_save + 'flu/new_number_picture_flu'
            tosave_nb_index_picture = self.to_save + 'nb_index_picture_flu'

            np.save(tosave_index_picture, self.index_picture)
            np.save(tosave_number_picture, number_picture)
            np.save(tosave_new_number_picture, number_picture)
            np.save(tosave_nb_index_picture, self.nb_index_picture)

    # ------------------------------------------
    def reshape_mix(self, name_toload, name_matvariable, name_tosave):
        shape_field_lw = [0 for i in range(self.config.nb_set)]
        shape_field_lc = [0 for i in range(self.config.nb_set)]

        start_time = timeit.default_timer()

        toload_index_picture = self.config.global_path_save + 'flu/index_picture_flu' + '.npy'
        toload_number_picture = self.config.global_path_save + 'flu/number_picture_flu' + '.npy'
        toload_nb_index_picture = self.config.global_path_save + 'flu/nb_index_picture_flu' + '.npy'

        index_picture = np.load(toload_index_picture)
        number_picture = np.load(toload_number_picture)
        nb_index_picture = np.load(toload_nb_index_picture)

        for i in range(self.config.nb_set):
            shape_field = np.load(self.config.global_path_load_raw \
                                  % (self.config.date[i], self.config.nexp[i],
                                     self.config.version_raw) + '/pict_event_flu/'
                                  + name_tosave + '_shape' + '.npy')

            shape_field_lw[i] = shape_field[0]
            shape_field_lc[i] = shape_field[1]

        stop_time = timeit.default_timer()
        print('tps pour trouver shape commune :', stop_time - start_time)

        minshape_lw = np.min(shape_field_lw)
        minshape_lc = np.min(shape_field_lc)
        maxshape_lw = np.max(shape_field_lw)
        maxshape_lc = np.max(shape_field_lc)

        nb_pict_so_far = 0
        new_number_picture = np.ones_like(number_picture) * np.nan
        numbertot_picture = np.ones_like(number_picture) * np.nan

        for i in range(self.config.nb_set):
            to_load_data = self.config.global_path_load_raw \
                           % (self.config.date[i], self.config.nexp[i], self.config.version_raw) + '/pict/'

            field = self.import_data(to_load_data, name_toload, name_matvariable)

            field, new_number_picture, numbertot_picture, nb_pict_set = self.reshape(self.sub_cycles[i],
                                                                                     new_number_picture,
                                                                                     field,
                                                                                     index_picture, number_picture,
                                                                                     numbertot_picture,
                                                                                     nb_pict_so_far)

            nb_pict_so_far = nb_pict_so_far + nb_pict_set

            print(nb_index_picture[i], nb_pict_set)
            print('verif reshape img dans set : nb_index_picture - nb_picture_set = {}'.format(
                nb_index_picture[i] - nb_pict_set))

            field = self.cut(field, minshape_lw, minshape_lc, maxshape_lw, maxshape_lc)

            shape = Shape(field)

            ## save
            if self.saving_step:
                tosave_field = self.to_save + name_tosave + '_{}'.format(i)
                tosave_shape_field = self.to_save + name_tosave + '_shape' + '_{}'.format(i)

                np.save(tosave_field, field)
                np.save(tosave_shape_field, np.array([shape.size_w, shape.size_c, shape.nb_pict]))

        # print('verif reshape img tot : nbtot_index_picture - nb_picture_son_far = {}'.format(
        #     np.sum(nb_index_picture) - nb_pict_so_far))

        return index_picture, new_number_picture, numbertot_picture, nb_index_picture

    # ------------------------------------------
    def reshape(self, sub_c, new_number_picture, field, index_picture, number_picture,
                numbertot_picture=None, nb_pict_so_far=None):

        start_time = timeit.default_timer()

        # print('sub_c', sub_c, np.size(sub_c))
        # print('shape field', np.shape(field))
        sub_index = index_picture[sub_c, :]
        sub_number = number_picture[sub_c, :]
        sub_nb_index = np.where(sub_index.reshape((sub_index.shape[0] * sub_index.shape[1])) == 1)[0].size

        # print('index size', np.shape(sub_index))
        # print_number(None, sub_number)
        # print('nb_cycle ds set', sub_index.shape[0])
        which_pict = np.zeros(sub_nb_index, dtype=int)

        k = 0
        for l in range(np.size(sub_c)):
            # print('cycle dans set ', l, 'nb_pict_ds_set_so_far', k)

            where_pict = np.where(sub_index[l, :] == 1)[0]
            # print('taille where pict dans cycle ', which_pict[k:k + where_pict.size].size)
            # print('taille number to keep dans cycle ', sub_number[l, where_pict].size)

            which_pict[k:k + where_pict.size] = sub_number[l, where_pict]

            # print('sur new_number : cycle = ', ini + l, 'ini = ', k,
            #       'fin = ', k + where_pict.size)

            new_number = np.arange(k, k + where_pict.size)
            new_number_picture[sub_c[l], where_pict] = new_number

            if self.config.mix_set:
                new_numbertot = np.arange(nb_pict_so_far + k, nb_pict_so_far + k + where_pict.size)
                numbertot_picture[sub_c[l], where_pict] = new_numbertot
            else:
                numbertot_picture = None
            k = k + where_pict.size

        # print_number(None, new_number_picture)
        # print_number(None, numbertot_picture)

        # print(which_pict, type(which_pict))
        nb_pict_set = k
        # print(k)
        field = field[:, :, which_pict]

        return field, new_number_picture, numbertot_picture, nb_pict_set


# ------------------------------------------
class Correlations():
    ''' rescale fluctuations autour de la moyenne '''

    # ---------------------------------------------------------#
    def __init__(self, config, f, ext, t, signaltype, NN_data, display_figure=False, saving_step=True):
        ''' display_figure : affiche les figures pour vérifier pas de pb dans l'analyse
            saving step : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas'''

        ## Config
        self.config = config

        self.f = f
        self.ext = ext
        self.t = t
        self.NN_data = NN_data
        self.NN_data = NN_data
        if self.NN_data != '':
            self.signaltype = signaltype + '_NN'
            self.fname = signaltype + '_' + self.NN_data
            self.savename = '_' + self.NN_data
        else:
            self.signaltype = signaltype
            self.fname = signaltype
            self.savename = ''

        self.nb_process = config.nb_process
        self.display_figure = display_figure
        self.saving_step = saving_step

        self.fraction = self.config.config_corr()

        self.to_load = self.config.global_path_save + '/' + self.signaltype + '/'
        self.to_save = self.config.global_path_save + '/' + self.signaltype + '/'
        self.to_save_fig = self.config.global_path_save + '/figure_' + self.signaltype + '/'

        if self.signaltype == signaltype and not config.mix_set:
            self.nbcycle = self.config.nbcycle
        elif self.signaltype == signaltype and config.mix_set:
            self.nbcycle = np.sum(self.config.nbcycle)
            self.sub_cycles = config.NN_sub_cycles
        elif self.signaltype == signaltype + '_NN' and not config.mix_set:
            self.nbcycle = np.load(self.to_load + self.fname + '_size.npy')
            self.cycle = np.load(self.to_load + self.fname + '_cycles.npy')
        else:
            self.nbcycle = np.load(self.to_load + self.fname + '_size.npy')
            recup_sub_cycles = Cell(self.to_load + self.fname + '_sub_cycles', config.nb_set)
            self.sub_cycles = recup_sub_cycles.reco_cell()

        self.eps_max = int(np.round(1 / self.fraction * np.size(self.t[0, :])))
        self.C_c, self.eps_C_c, self.plateau_c = self.correlations_par_cycle()
        self.stats_plateau_c = Stat(self.config, self.plateau_c)

        self.mean_C, self.eps_mean_C, self.plateau_mean_C, self.inv_mean_C = self.mean_correlations()

        ## save
        if saving_step:
            self.tosave_C_c = self.to_save + 'C_c' + self.savename
            self.tosave_eps_C_c = self.to_save + 'eps_C_c' + self.savename
            self.tosave_plateau_c = self.to_save + 'plateau_c' + self.savename
            self.tosave_mean_C = self.to_save + 'mean_C' + self.savename
            self.tosave_eps_mean_C = self.to_save + 'eps_mean_C' + self.savename
            self.tosave_plateau_mean_C = self.to_save + 'plateau_mean_C' + self.savename
            self.tosave_inv_mean_C = self.to_save + 'inv_mean_C' + self.savename

            np.save(self.tosave_C_c, self.C_c)
            np.save(self.tosave_eps_C_c, self.eps_C_c)
            np.save(self.tosave_plateau_c, self.plateau_c)
            np.save(self.tosave_mean_C, self.mean_C)
            np.save(self.tosave_eps_mean_C, self.eps_mean_C)
            np.save(self.tosave_plateau_mean_C, self.plateau_mean_C)
            np.save(self.tosave_inv_mean_C, self.inv_mean_C)

    # ------------------------------------------
    def correlations_par_cycle(self):

        eps = self.ext[:, 0:self.eps_max]

        C = np.zeros((self.nbcycle, self.eps_max))
        plateaux = np.zeros(self.nbcycle)

        for i in range(self.nbcycle):
            fluctuations = self.f[i, :]
            c = np.zeros(self.eps_max)
            c[0] = 0
            for j in range(1, self.eps_max):
                c[j] = 1 / 2 * np.mean((fluctuations[j::] - fluctuations[0:-j]) ** 2)

            plateaux[i] = np.mean(c[int(np.round(self.eps_max / 3))::])
            C[i, :] = c

        return C, eps, plateaux

    # ------------------------------------------
    def mean_correlations(self):

        C = np.zeros(self.eps_max)
        eps_C = np.zeros(self.eps_max)

        for i in range(self.eps_max):
            C[i] = np.mean(self.C_c[:, i])
            eps_C[i] = np.mean(self.eps_C_c[:, i])

        plateau = np.mean(C[int(np.round(self.eps_max / 3))::])
        inv_C = -C + plateau

        return C, eps_C, plateau, inv_C
