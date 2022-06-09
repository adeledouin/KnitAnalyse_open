import numpy as np
from functools import partial
from multiprocessing import Pool, Array
import ctypes
import timeit
from pathlib import Path
from skimage import measure

from Utils.classFindPeak import Derivee, FindPeak
from Utils.classCell import Cell
from Utils.classStat import Stat, Shape
import memory
import logging
from dictdata import dictdata

logging.basicConfig(format='| %(levelname)s | %(asctime)s | %(message)s', level=logging.INFO)


def def_names(signaltype, fname, NN_data, Sm=None):
    """
    Function to define folders and files names

    Parameters:
        signaltype (str) : 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence'
        fname (str) : None if pas différent de signaltype
        NN_data (str) : '', 'train', 'val', 'test'
        Sm (bol): rsc par Sm (2nd moment) du signal des events

    Returns:
        output (str) : nom du dossier, nom du fichier force avec extension NN, extention Sm pour df_tt, extension Sm pour df_seuil
    """

    if NN_data != '':
        if fname is None:
            fname = signaltype + '_' + NN_data
        else:
            fname = fname + '_' + NN_data
        signaltype = signaltype + '_NN'
        savename = '_' + NN_data
    else:
        if fname is None:
            fname = signaltype
        else:
            fname = fname
        signaltype = signaltype
        savename = ''
    if Sm is not None:
        if Sm:
            savename_df_tt = '_Sm_tt' + savename
            savename_df_seuil = '_Sm_seuil' + savename
        else:
            savename_df_tt = '_tt' + savename
            savename_df_seuil = '_seuil' + savename
    else:
        savename_df_tt = None
        savename_df_seuil = None


    return signaltype, fname, savename, savename_df_tt, savename_df_seuil

def def_nbcycle(config, path_signal, fname, NN_data):
    """
    Function to define number of cycle on the actual signal

    Parameters:
        config (class) : config associée à la l'analyse
        path_signal (str) : path to folder
        fname (str) : nomd du fichier
        NN_data (str) : '', 'train', 'val', 'test'

    Returns:
        output : nbcycle, sub_nbcycles, cycles, sub_cycles_NN, NN_sub_cycles
        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse
        cycles (list): liste des cycles dans cette extention NN (not None seulement quand NN et single set)
        sub_cycles_NN (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre total de cycles dans analyse
        NN_sub_cycles (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre de cycles total dans NN
    """

    if NN_data == '' and not config.mix_set:
        nbcycle = config.nbcycle
        sub_cycles = None
        cycles = None
        sub_cycles_NN = None
        NN_sub_cycles = None
    elif NN_data == '' and config.mix_set:
        nbcycle = np.sum(config.nbcycle)
        sub_cycles = config.sub_cycles
        cycles = None
        sub_cycles_NN = None
        NN_sub_cycles = None
    elif NN_data != '' and not config.mix_set:
        nbcycle = np.load(path_signal + fname + '_size.npy')
        sub_cycles = None
        cycles = np.load(path_signal + fname + '_cycles.npy')
        sub_cycles_NN = None
        NN_sub_cycles = None
    else:
        nbcycle = np.load(path_signal + fname + '_size.npy')
        sub_cycles = config.sub_cycles
        cycles = None
        recup_sub_cycles_old = Cell(path_signal + fname + '_sub_cycles_NN', config.nb_set)
        sub_cycles_NN = recup_sub_cycles_old.reco_cell()
        recup_sub_cycles = Cell(path_signal + fname + '_NN_sub_cycles', config.nb_set)
        NN_sub_cycles = recup_sub_cycles.reco_cell()

    return nbcycle, cycles, sub_cycles, sub_cycles_NN, NN_sub_cycles


# ---------------------------------------------------------------------------------------------------------------------#
class ForceEvent():
    """
    Classe qui permet de trouver les events dans le signal en force.

    Attributes:
        config (class) : config associée à la l'analyse

        signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' nom du dossier
        fname (str) : nom du fichier du sigal en force - None if pas différent de signaltype
        NN_data (str) : '', 'train', 'val', 'test' extention NN
        Sm (bol) : rsc par Sm (2nd moment) du signal des events
        savename_df_tt (str) : extension de rsc de df tt
        savename_df_seuil : extension de rsc de df_seuil

        nb_process (int) : no de process a utiliser pour le mutliprocess
        saving step (bol) : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas
        display_figure (bol) : affiche les figure pendant recherche des events si besoin

        path_signal (str) : chemin du dossier associé à ce signal
        to_save_fig (str) : chemin associé pour save fig

        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse
        cycles (list): liste des cycles dans cette extention NN (not None seulement quand NN et single set)
        sub_cycles_NN (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre total de cycles dans analyse
        NN_sub_cycles (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre de cycles total dans NN

        f (array) : signal force
        t (array) : tps associé au signal
        ext (array) : extension associé au signal
        f_size (int) : taille d'un cycle

        df_tt (array) : array 1D des events
        dext_tt (array) : array 1D des extension associé aux events
        dt_tt (array) : array 1D  du tps associé aux events
        index_tt (array) : index des events associées au signal
        number_tt (array) : numero des events associées au signal
        nb_index_tt (int ou array) : nombre total d'events associées au signal
    """

    # ---------------------------------------------------------#
    def __init__(self, config, f, ext, t, signaltype, NN_data, fname=None, Sm=False, display_figure_debug=False,
                 saving_step=True):
        """
        The constructor for ForceEvent.

        Parameters:
            config (class) : config associée à la l'analyse

            f (array) : signal force
            t (array) : tps associé au signal
            ext (array) : extension associé au signal

            signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force
            NN_data (str) : '', 'train', 'val', 'test' extension NN
            fname (str) : nom du signal - None if pas différent de signaltype

            Sm (bol) : rsc par Sm (2nd moment) du signal des events

            display_figure_debug (bol) : affiche les figure pendant recherche des events si besoin
            saving step (bol) : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas
        """

        ## Config
        self.config = config

        self.f = f
        self.ext = ext
        self.t = t

        self.NN_data = NN_data
        self.Sm = Sm
        self.signaltype, self.fname, _, self.savename_df_tt, self.savename_df_seuil = def_names(signaltype, fname, NN_data, Sm)

        self.nb_process = config.nb_process
        self.display_figure = display_figure_debug
        self.saving_step = saving_step

        self.path_signal = self.config.global_path_save + '/' + self.signaltype + '/'
        self.to_save_fig = self.config.global_path_save + '/figure_' + self.signaltype + '/'

        self.nbcycle, self.cycles, self.sub_cycles, self.sub_cycles_NN, self.NN_sub_cycles = def_nbcycle(self.config,
                                                                                                            self.path_signal,
                                                                                                            self.fname,
                                                                                                            self.NN_data)


        ## events info

        self.f_size = np.size(self.f[0, :])

        self.df_tt, self.dt_tt, self.dext_tt, self.index_df_tt, self.number_df_tt, self.nb_df_tt, \
        self.min_indice_df_tt, self.max_indice_df_tt = self.df_tt()

    # ------------------------------------------
    def import_single(self, name, extension='npy', size=None):

        if extension == 'npy':
            to_load = self.path_signal + name + '.npy'
            single = np.load(to_load)
        else:
            recup_single = Cell(self.path_signal + name, size)
            single = recup_single.reco_cell()

        return single

    # ------------------------------------------
    def save_single(self, path_signal, data, name, extension='npy', nbfichier=None):

        if data is not None:
            if extension == 'npy':
                to_save = path_signal + name
                np.save(to_save, data)
            elif extension == 'cell':
                Cell(path_signal + name, nbfichier, data=data, extension='cell')
            else:
                Cell(path_signal + name, nbfichier, data=data, extension='csv')

    # ------------------------------------------
    def find_indice_event(self):
        ''' '''

        min_indice = [0 for i in range(self.nbcycle)]
        max_indice = [0 for i in range(self.nbcycle)]
        min_indice_size = np.zeros(self.nbcycle, dtype=int)
        max_indice_size = np.zeros(self.nbcycle, dtype=int)

        for i in range(self.nbcycle):

            der = Derivee(self.f[i, :], self.ext[i, :], self.t[i, :])

            findpeak = FindPeak(der.der_signe_der_f, i, brut_signal=False)

            min, max, min_size, max_size = findpeak.recup_min_max_indices()

            min_indice[i] = min
            max_indice[i] = max
            min_indice_size[i] = min_size
            max_indice_size[i] = max_size

        return min_indice, max_indice, min_indice_size, max_indice_size

    # ------------------------------------------
    def find_event(self, plot=None):

        if self.signaltype == 'm_f':
            index_event = [0 for i in range(self.nbcycle)]
            number_event = [0 for i in range(self.nbcycle)]
        else:
            index_event = np.zeros((self.nbcycle, self.f_size))
            number_event = np.ones((self.nbcycle, self.f_size)) * np.nan

        nb_event = 0

        k = 0
        for i in range(self.nbcycle):
            for j in range(self.min_indice_size[i]):
                index_event[i, 1 + self.max_indice[i][j]:2 + self.min_indice[i][j]] = 1
                number_event[i, 1 + self.max_indice[i][j]:2 + self.min_indice[i][j]] = k + j

            nb_event = nb_event + self.min_indice_size[i]
            k = k + self.min_indice_size[i] - 1

        return index_event, number_event, nb_event

    # ------------------------------------------
    def ampli_events(self):

        nb_df_tt = self.nb_events
        df_tt = np.zeros(nb_df_tt)
        dt_tt = np.zeros(nb_df_tt)
        dext_tt = np.zeros(nb_df_tt)
        min_indice_df_tt = np.zeros((2, nb_df_tt))
        max_indice_df_tt = np.zeros((2, nb_df_tt))

        index_df_tt = np.zeros((self.nbcycle, self.f_size))
        number_df_tt = np.ones((self.nbcycle, self.f_size)) * np.nan

        k = 0
        for i in range(self.nbcycle):

            for j in range(self.min_indice_size[i]):
                index_df_tt[i, 1 + self.max_indice[i][j]] = 1
                number_df_tt[i, 1 + self.max_indice[i][j]] = int(k + j)
                df_tt[k + j] = self.f[i, 1 + self.max_indice[i][j]] - self.f[i, 1 + self.min_indice[i][j]]
                dt_tt[k + j] = np.abs(self.t[i, 1 + self.max_indice[i][j]] - self.t[i, 1 + self.min_indice[i][j]])
                dext_tt[k + j] = self.ext[i, 1 + self.max_indice[i][j]] - self.ext[i, 1 + self.min_indice[i][j]]

                min_indice_df_tt[0, k + j] = i
                min_indice_df_tt[1, k + j] = 1 + self.min_indice[i][j]
                max_indice_df_tt[0, k + j] = i
                max_indice_df_tt[1, k + j] = 1 + self.max_indice[i][j]

                # if j == 0:
                #     m_tt[k + j] = (self.f[i, 1 + self.max_indice[i][j]] - self.f[i, 0]) / \
                #                   (self.t[i, 1 + self.max_indice[i][j]] - self.t[i, 0])
                # else:
                #     m_tt[k + j] = (self.f[i, 1 + self.max_indice[i][j]] - self.f[i, 1 + self.min_indice[i][j-1]]) / \
                #                   (self.t[i, 1 + self.max_indice[i][j]] - self.t[i, 1 + self.min_indice[i][j-1]])

            k = k + self.min_indice_size[i]

        return df_tt, dt_tt, dext_tt, index_df_tt, number_df_tt, nb_df_tt, min_indice_df_tt, max_indice_df_tt

    # ------------------------------------------
    def df_tt(self):

        ## regarde si le fichier existe dejà :
        fileName = self.path_signal + 'df' + self.savename_df_tt + '.npy'
        fileObj = Path(fileName)

        is_fileObj = fileObj.is_file()

        if is_fileObj:
            print('df_tt déjà enregisté')

            df_tt = self.import_single('df' + self.savename_df_tt)
            dext_tt = self.import_single('dext' + self.savename_df_tt)
            dt_tt = self.import_single('dt' + self.savename_df_tt)

            index_df_tt = self.import_single('index_df' + self.savename_df_tt)
            number_df_tt = self.import_single('number_df' + self.savename_df_tt)
            nb_index_df_tt = self.import_single('nb_df' + self.savename_df_tt)

            min_indice_df_tt = self.import_single('min_indice_df' + self.savename_df_tt)
            max_indice_df_tt = self.import_single('max_indice_df' + self.savename_df_tt)

            self.min_indice = self.import_single('min_indice', extension='cell', size=self.nbcycle)
            self.max_indice = self.import_single('max_indice', extension='cell', size=self.nbcycle)

            self.min_indice_size = self.import_single('min_indice_size')
            self.max_indice_size = self.import_single('max_indice_size')

            self.index_events = self.import_single('index_events')
            self.number_events = self.import_single('number_events')
            self.nb_index_events = self.import_single('nb_events')

        else:
            print('df_tt non enregisté')
            self.min_indice, self.max_indice, self.min_indice_size, self.max_indice_size = self.find_indice_event()

            self.index_events, self.number_events, self.nb_events = self.find_event()

            df_tt, dt_tt, dext_tt, index_df_tt, number_df_tt, nb_index_df_tt, \
            min_indice_df_tt, max_indice_df_tt = self.ampli_events()

            if self.Sm:
                stats = Stat(self.config, df_tt)
                df_tt = df_tt / stats.m2

            if self.saving_step:

                self.save_single(self.path_signal, self.min_indice, 'min_indice', extension='cell',
                                 nbfichier=self.nbcycle)
                self.save_single(self.path_signal, self.max_indice, 'max_indice', extension='cell',
                                 nbfichier=self.nbcycle)

                self.save_single(self.path_signal, self.min_indice_size, 'min_indice_size')
                self.save_single(self.path_signal, self.max_indice_size, 'max_indice_size')

                self.save_single(self.path_signal, self.index_events, 'index_events')
                self.save_single(self.path_signal, self.number_events, 'number_events')
                self.save_single(self.path_signal, self.nb_events, 'nb_events')

                self.save_single(self.path_signal, df_tt, 'df' + self.savename_df_tt)
                self.save_single(self.path_signal, dext_tt, 'dext' + self.savename_df_tt)
                self.save_single(self.path_signal, dt_tt, 'dt' + self.savename_df_tt)
                self.save_single(self.path_signal, nb_index_df_tt, 'nb_df' + self.savename_df_tt)

                self.save_single(self.path_signal, min_indice_df_tt, 'min_indice_df' + self.savename_df_tt)
                self.save_single(self.path_signal, max_indice_df_tt, 'max_indice_df' + self.savename_df_tt)

                if self.signaltype == 'm_f':
                    tosave_index_df_tt = self.path_signal + 'index_df' + self.savename_df_tt
                    tosave_number_df_tt = self.path_signal + 'number_df' + self.savename_df_tt
                    Cell(tosave_index_df_tt, self.nbcycle, index_df_tt, 'cell')
                    Cell(tosave_number_df_tt, self.nbcycle, number_df_tt, 'cell')
                else:
                    self.save_single(self.path_signal, index_df_tt, 'index_df' + self.savename_df_tt)
                    self.save_single(self.path_signal, number_df_tt, 'number_df' + self.savename_df_tt)

        return df_tt, dt_tt, dext_tt, index_df_tt, number_df_tt, nb_index_df_tt, min_indice_df_tt, max_indice_df_tt

    # ------------------------------------------
    def df_tab(self):

        df_tab = np.zeros_like(self.index_df_tt)

        where_df = np.where(self.index_df_tt == 1)
        for i in range(where_df[0].size):
            df_tab[where_df[0][i], where_df[1][i]] = self.df_tt[i]

        return df_tab

    # ------------------------------------------
    def df_seuil(self, min, max, df_tab):

        if max is None:
            where = np.where(self.df_tt >= min)[0]
        else:
            where = np.where((self.df_tt >= min) & (self.df_tt < max))[0]
        df_seuil = self.df_tt[where]
        dt_seuil = self.dt_tt[where]

        if max is None:
            where = np.where(df_tab >= min)
        else:
            if min == 0:
                where = np.where((df_tab > min) & (df_tab < max))
            else:
                where = np.where((df_tab >= min) & (df_tab < max))
        index_df_seuil = np.zeros_like(self.index_df_tt)
        number_df_seuil = np.zeros_like(self.index_df_tt)

        new_number = np.arange(1, where[0].size + 1)

        for i in range(where[0].size):
            index_df_seuil[where[0][i], where[1][i]] = 1
            number_df_seuil[where[0][i], where[1][i]] = new_number[i]

        return df_seuil, dt_seuil, index_df_seuil, number_df_seuil

    # ------------------------------------------
    def time_btw_df(self, index_signal, number_signal):

        t_btw = np.array([0])
        index_t_btw = index_signal
        number_t_btw = number_signal

        for i in range(self.nbcycle):
            where_df = np.where(index_signal[i, :] == 1)[0]
            if where_df.size != 0:
                t_btw_sub = self.t[i, where_df[1:]] - self.t[i, where_df[:-1]]

                index_t_btw[i, where_df[-1]] = 0
                number_t_btw[i, where_df] = number_t_btw[i, where_df] - i
                number_t_btw[i, where_df[-1]] = np.nan
            else:
                t_btw_sub = []

            t_btw = np.hstack((t_btw, t_btw_sub))

        t_btw = t_btw[1:]

        return t_btw, index_t_btw, number_t_btw


