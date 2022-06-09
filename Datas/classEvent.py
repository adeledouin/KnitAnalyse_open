import numpy as np
from functools import partial
from multiprocessing import Pool, Array
import ctypes
import timeit
from pathlib import Path
from skimage import measure
import logging

from Utils.classFindPeak import Derivee, FindPeak
from Utils.classCell import Cell
from Utils.classStat import Stat, Shape
from dictdata import dictdata
import memory

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

def fname_to_fsave(fname, sep_posneg, signe):
    """
    Function to compute class ForceEvent on the right field

    Parameters:


    Returns:
        output :
    """

    if fname == 'dev':
        fsave = '_dev'
    elif fname == 'vort' and not sep_posneg:
        fsave ='_vort'
    elif fname == 'vort' and sep_posneg:
        if signe == 'pos':
            fsave = '_vort_p'
        elif signe == 'neg':
            fsave = '_vort_n'
        else:
            fsave = '_vort_pn'
    else:
        if signe == 'pos':
            fsave = '_slip_p'
        elif signe == 'neg':
            fsave = '_slip_n'
        else:
            fsave = '_slip_pn'

    return fsave

def fsave_to_fname(fsave):
    """
    Function to compute class ForceEvent on the right field

    Parameters:


    Returns:
        output :
    """

    # print(fsave)
    if fsave == '_dev':
        fname = 'dev'
        sep_posneg = False
        signe =''
    elif fsave == '_vort':
        fname = 'vort'
        sep_posneg = False
        signe = ''
    elif fsave == '_vort_p':
        fname = 'vort'
        sep_posneg = True
        signe = 'pos'
    elif fsave == '_vort_n':
        fname = 'vort'
        sep_posneg = True
        signe = 'neg'
    elif fsave == '_vort_pn':
        fname = 'vort'
        sep_posneg = True
        signe = 'cumul'
    elif fsave == '_slip_p':
        fname = 'slip_Y'
        sep_posneg = False
        signe = 'pos'
    elif fsave == '_slip_n':
        fname = 'slip_Y'
        sep_posneg = False
        signe = 'neg'
    elif fsave == '_slip_pn':
        fname = 'slip_Y'
        sep_posneg = False
        signe = 'cumul'

    return fname, sep_posneg, signe

def imgevent_cumul_pn(data_p, data_n, concat=False):
    """
    Function to compute class ForceEvent on the right field

    Parameters:


    Returns:
        output :
    """

    if not concat:
        data = np.zeros_like(data_p)
        for i in range(np.size(data)):
            data[i] = data_p[i] + data_n[i]
    else:
        data = np.concatenate((data_p, data_n))

    return data

def find_img_events(config, signal_img, fsave, fname, seuil, save_seuil, signaltype, NN_data, sep_posneg, signe):
    """
    Function to compute class ForceEvent on the right field

    Parameters:


    Returns:
        output :
    """

    if fsave == '_dev':
        event = StatsImgEvent(config, signal_img.dev, seuil, save_seuil,
                              signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        nb_area_img = event.nb_area_img
        sum_S_a = event.sum_S_a
        sum_S_f = event.sum_S_f
        S_a = event.S_a
        S_f = event.S_f
        pict_S = event.pict_S

    elif fsave == '_vort':
        event = StatsImgEvent(config, signal_img.vort, seuil, save_seuil,
                              signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        nb_area_img = event.nb_area_img
        sum_S_a = event.sum_S_a
        sum_S_f = event.sum_S_f
        S_a = event.S_a
        S_f = event.S_f
        pict_S = event.pict_S

    elif fname == 'vort' and sep_posneg:
        event = StatsImgEvent(config, signal_img.vort, seuil, save_seuil,
                              signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        if signe == 'pos':
            nb_area_img = event.nb_area_img_p
            sum_S_a = event.sum_S_a_p
            sum_S_f = event.sum_S_f_p
            S_a = event.S_a_p
            S_f = event.S_f_p
            pict_S = event.pict_S

        elif signe == 'neg':
            nb_area_img = event.nb_area_img_n
            sum_S_a = event.sum_S_a_n
            sum_S_f = event.sum_S_f_n
            S_a = event.S_a_n
            S_f = event.S_f_n
            pict_S = event.pict_S

        else:
            nb_area_img = imgevent_cumul_pn(event.nb_area_img_p, event.nb_area_img_n)
            sum_S_a = imgevent_cumul_pn(event.sum_S_a_p, event.sum_S_a_n)
            sum_S_f = imgevent_cumul_pn(event.sum_S_f_p, event.sum_S_f_n)
            S_a = imgevent_cumul_pn(event.S_a_p, event.S_a_n, concat=True)
            S_f = imgevent_cumul_pn(event.S_f_p, event.S_f_n, concat=True)
            pict_S = imgevent_cumul_pn(event.pict_S_p, event.pict_S_n, concat=True)

    else:
        event_p = StatsImgEvent(config, signal_img.slip_Y, seuil, save_seuil,
                                signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        event_n = StatsImgEvent(config, signal_img.slip_X, seuil, save_seuil,
                                signaltype=signaltype, NN_data=NN_data, fname=fname, sep_posneg=sep_posneg)

        if signe == 'pos':
            nb_area_img = event_p.nb_area_img
            sum_S_a = event_p.sum_S_a
            sum_S_f = event_p.sum_S_f
            S_a = event_p.S_a
            S_f = event_p.S_f
            pict_S = event_p.pict_S

        elif signe == 'neg':
            nb_area_img = event_n.nb_area_img
            sum_S_a = event_n.sum_S_a
            sum_S_f = event_n.sum_S_f
            S_a = event_n.S_a
            S_f = event_n.S_f
            pict_S = event_n.pict_S

        else:
            nb_area_img = imgevent_cumul_pn(event_p.nb_area_img, event_n.nb_area_img)
            sum_S_a = imgevent_cumul_pn(event_p.sum_S_a, event_n.sum_S_a)
            sum_S_f = imgevent_cumul_pn(event_p.sum_S_f, event_n.sum_S_f)
            S_a = imgevent_cumul_pn(event_p.S_a, event_n.S_a, concat=True)
            S_f = imgevent_cumul_pn(event_p.S_f, event_n.S_f, concat=True)
            pict_S = imgevent_cumul_pn(event_p.pict_S, event_n.pict_S, concat=True)

    return nb_area_img, sum_S_a, sum_S_f, S_a, S_f, pict_S

def find_img_events_new(config, signal_img, signaltype, NN_data, pxmm, which_field, seuil, add_sub='', num_set=None):
    """
    Function to compute class ForceEvent on the right field

    Parameters:


    Returns:
        output :
    """

    if which_field == '_vort':
        vort_p_event = FaultImgEvent(config, pxmm, signal_img.vort.copy(), signal_img.abs_vort,
                                     signal_img.posx, signal_img.posy, signal_img.posX, signal_img.posY,
                                     seuil, signaltype=signaltype, fname='vort{}'.format(add_sub), NN_data=NN_data,
                                     signe='_p', num_set=num_set)
        vort_n_event = FaultImgEvent(config, pxmm, -signal_img.vort.copy(), signal_img.abs_vort,
                                     signal_img.posx, signal_img.posy, signal_img.posX, signal_img.posY,
                                     seuil, signaltype=signaltype, fname='vort{}'.format(add_sub), NN_data=NN_data,
                                     signe='_n', num_set=num_set)

        fault_event = [vort_n_event, vort_p_event]
        nb_area_img = fault_event[0].info_stats['nb_area_img'] + fault_event[1].info_stats['nb_area_img']
        sum_S_a = fault_event[0].info_stats['sum_S_a'] + fault_event[1].info_stats['sum_S_a']
        sum_S_f = fault_event[0].info_stats['sum_S_f'] + fault_event[1].info_stats['sum_S_f']
        sum_S_nf = fault_event[0].info_stats['sum_S_nf'] + fault_event[1].info_stats['sum_S_nf']
        S_a = np.concatenate((fault_event[0].info_stats['S_a'], fault_event[1].info_stats['S_a']))
        S_f = np.concatenate((fault_event[0].info_stats['S_f'], fault_event[1].info_stats['S_f']))
        S_nf = np.concatenate((fault_event[0].info_stats['S_nf'], fault_event[1].info_stats['S_nf']))
        pict_S = np.concatenate((fault_event[0].info_stats['pict_S'], fault_event[1].info_stats['pict_S']))

    else:
        if which_field == '_vort_pn':
            fault_event = FaultImgEvent(config, pxmm, signal_img.vort.copy(), signal_img.abs_vort,
                                         signal_img.posx, signal_img.posy, signal_img.posX, signal_img.posY,
                                         seuil, signaltype=signaltype, fname='vort{}'.format(add_sub), NN_data=NN_data,
                                         signe='_pn', num_set=num_set)

        elif which_field == '_dev':
            fault_event = FaultImgEvent(config, pxmm, signal_img.dev.copy(), signal_img.dev,
                                      signal_img.posx, signal_img.posy, signal_img.posX, signal_img.posY,
                                      seuil, signaltype=signaltype, fname='dev{}'.format(add_sub), NN_data=NN_data, signe='',
                                      num_set=num_set)


        elif which_field == '_slip_Y':
            fault_event = FaultImgEvent(config, pxmm, np.abs(signal_img.slip_Y.copy()), signal_img.slip,
                                         signal_img.posx, signal_img.posy, signal_img.posX, signal_img.posY,
                                         seuil, signaltype=signaltype, fname='slip_Y{}'.format(add_sub), NN_data=NN_data,
                                         signe='', num_set=num_set)

        elif which_field == '_slip_X':
            fault_event = FaultImgEvent(config, pxmm, np.abs(signal_img.slip_X.copy()), signal_img.slip,
                                         signal_img.posx, signal_img.posy, signal_img.posX, signal_img.posY,
                                         seuil, signaltype=signaltype, fname='slip_X{}'.format(add_sub), NN_data=NN_data,
                                         signe='', num_set=num_set)

        nb_area_img = fault_event.info_stats['nb_area_img']
        sum_S_a = fault_event.info_stats['sum_S_a']
        sum_S_f = fault_event.info_stats['sum_S_f']
        sum_S_nf = fault_event.info_stats['sum_S_nf']
        S_a = fault_event.info_stats['S_a']
        S_f = fault_event.info_stats['S_f']
        S_nf = fault_event.info_stats['S_nf']
        pict_S = fault_event.info_stats['pict_S']

    return fault_event, nb_area_img, sum_S_a, sum_S_f, S_a, S_f, pict_S


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
    def df_seuil_tab(self, df, index_df):

        df_tab = np.zeros_like(index_df)

        where_df = np.where(index_df == 1)
        for i in range(where_df[0].size):
            df_tab[where_df[0][i], where_df[1][i]] = df[i]

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
        number_df_seuil = np.ones_like(self.index_df_tt) * np.NaN

        new_number = np.arange(0, where[0].size)

        for i in range(where[0].size):
            index_df_seuil[where[0][i], where[1][i]] = 1
            number_df_seuil[where[0][i], where[1][i]] = new_number[i]

        return df_seuil, dt_seuil, index_df_seuil, number_df_seuil

    # ------------------------------------------
    def S_f_seuil(self, min, max, S_f, S_f_tab):

        if max is None:
            where = np.where(S_f >= min)[0]
        else:
            where = np.where((S_f >= min) & (S_f < max))[0]
        S_f_seuil = S_f[where]

        if max is None:
            where = np.where(S_f_tab >= min)
        else:
            if min == 0:
                where = np.where((S_f_tab > min) & (S_f_tab < max))
            else:
                where = np.where((S_f_tab >= min) & (S_f_tab < max))
        index_S_f_seuil = np.zeros_like(S_f_tab)
        number_S_f_seuil = np.ones_like(S_f_tab) * np.NaN

        new_number = np.arange(0, where[0].size)

        for i in range(where[0].size):
            index_S_f_seuil[where[0][i], where[1][i]] = 1
            number_S_f_seuil[where[0][i], where[1][i]] = new_number[i]

        return S_f_seuil, index_S_f_seuil, number_S_f_seuil

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

    # ------------------------------------------
    def time_bfr_df(self, index_signal):
        start_time = timeit.default_timer()

        t_bfr = np.zeros((self.nbcycle, self.f_size))

        for i in range(self.nbcycle):
            # prin(i)

            where = np.where(index_signal[i, :] == 1)[0]
            ref = np.concatenate((np.array([0]), where[:-1]))
            # remplire = [np.arange(where[j]-ref[j], 0, -1) for j in range(where.size)]
            for j in range(where.size):
                t_bfr[i, ref[j]:where[j]] = np.arange(where[j] - ref[j], 0, -1)

        stop_time = timeit.default_timer()
        print('tps pour calculer t_bfr:', stop_time - start_time)

        return t_bfr

    # ------------------------------------------
    def nb_df_btwpict(self, df_tab, index_picture):
        nb_df = []
        for i in range(df_tab.shape[0]):

            where = np.where(index_picture[i, :] == 1)[0]
            ref = np.concatenate((np.array([0]), where[:-1]))

            for j in range(where.size):
                if where[j] == 0:
                    nb_df.append(0)
                else:
                    sub_df = df_tab[i, ref[j]:where[j]]
                    nb_df.append(np.sum(sub_df != 0))

        return np.asarray(nb_df)

    # ------------------------------------------
    def df_btw_pict(self, df_tab, index_picture, number_picture, nb_df_btw):
        k = 0
        sum_df_btw_tab = np.zeros((df_tab.shape[0], self.f_size))
        sum_df_btw = np.zeros(nb_df_btw.size)
        max_df_btw_tab = np.zeros((df_tab.shape[0], self.f_size))
        max_df_btw = np.zeros(nb_df_btw.size)
        index_df_btw = index_picture
        number_df_btw = number_picture

        for i in range(df_tab.shape[0]):
            where = np.where(index_picture[i, :] == 1)[0]
            ref = np.concatenate((np.array([0]), where[:-1]))

            for j in range(where.size):
                if where[j] == 0:
                    sum_df_btw_tab[i, where[j]] = 0
                    sum_df_btw[k] = 0
                    max_df_btw_tab[i, where[j]] = 0
                    max_df_btw[k] = 0
                    k += 1
                else:
                    sub_df = df_tab[i, ref[j]:where[j]]
                    sum_df_btw_tab[i, where[j]] = np.sum(sub_df)
                    sum_df_btw[k] = np.sum(sub_df)
                    max_df_btw_tab[i, where[j]] = np.max(sub_df)
                    max_df_btw[k] = np.max(sub_df)
                    k += 1

        return sum_df_btw_tab, sum_df_btw, max_df_btw_tab, max_df_btw, index_df_btw, number_df_btw


# ---------------------------------------------------------------------------------------------------------------------#
class InfoField():
    """
    Classe qui permet de charger signal des event en force et ses dépendances.

    Attributes:
        config (class) : config associée à la l'analyse

        nb_area (int) : nombre de region dans l'img
        num_area (1D array) : numéro des regions
        size_area (1D array) : taille en nombre de mailles de chaque region
        sum_field (1D array) : sum des valeurs du champs sur les mailles d'une region, pour toutes les regions
        size_area_img (int) :  taille en nombre de mailles compté sur toutes les regions
        size_field_img (int) : sum des valeurs de champs sur toutes les regions
        conncomp (array) : Labeled array, where all connected regions are assigned the same integer value.

    """

    # ---------------------------------------------------------#
    def __init__(self, config, field, normfield, field_seuil, seuil, fname, signe, path, fault_analyse=True,
                 debug=False):
        """
        The constructor for InfoField.

        Parameters:
            config (class) : config associée à la l'analyse

            field (array) : une img de champ
            field_seuil (array) : labeled array, where all regions supp to seuil are assigned 1.
            seuil (int) : valeur utilisée pour seuiller les évents
            fault_analyse (bol) : est ce que l'analyse est pour étudier les fault
            debug (bol) : permet d'afficher plot des img des region pour debuguer

        """

        self.config = config

        self.fname = fname
        self.signe = signe
        self.path = path

        self.field = field
        self.normfield = normfield
        self.field_seuil = field_seuil
        self.seuil = seuil

        self.nb_area, self.num_area, self.size_area, self.center, self.orientation, self.sum_field, self.sum_normfield, \
        self.size_area_img, self.sum_field_img, self.sum_normfield_img, self.conncomp = self.info(fault_analyse,
                                                                                                  debug)

    # ------------------------------------------
    def info(self, fault_analyse, debug):

        info = dictdata()
        nb_area, num_area, size_area, center_area, orientation_area, sum_field, sum_normfield, \
        size_area_img, sum_field_img, sum_normfield_img, conncomp = self.analyse_info(
            fault_analyse,
            debug)
        info.add('nb_area', nb_area)
        info.add('num_area', num_area)
        info.add('size_area', size_area)
        info.add('center_area', center_area)
        info.add('orientation_area', orientation_area)
        info.add('sum_field', sum_field)
        info.add('sum_normfield', sum_normfield)
        info.add('size_area_img', size_area_img)
        info.add('sum_field_img', sum_field_img)
        info.add('sum_normfield_img', sum_normfield_img)
        info.add('conncomp', conncomp)

        return nb_area, num_area, size_area, center_area, orientation_area, sum_field, sum_normfield, \
               size_area_img, sum_field_img, sum_normfield_img, conncomp

    # ------------------------------------------
    def analyse_info(self, fault_analyse, debug):

        start_time = timeit.default_timer()

        conncomp, Nobj = measure.label(self.field_seuil, return_num=True)

        Reg = measure.regionprops(conncomp)

        stop_time = timeit.default_timer()
        # print('tps pour regionprop :', stop_time - start_time)

        start_time = timeit.default_timer()

        num_area = np.arange(1, Nobj + 1)
        Area = np.zeros(Nobj)
        Center = np.zeros((Nobj, 2))
        Orient = np.zeros(Nobj)

        for i in range(Nobj):
            Area[i] = Reg[i].area
            if Area[i] <= 1:
                pixels = np.nonzero(conncomp == num_area[i])
                conncomp[pixels] = 0
            else:
                Center[i, :] = Reg[i].centroid
                Orient[i] = Reg[i].orientation

        num_area = num_area[np.where(Area > 1)[0]]
        center = Center[np.where(Area > 1)[0], :]
        orientation = Orient[np.where(Area > 1)[0]]
        Area = Area[np.where(Area > 1)[0]]

        Nobj = np.size(Area)

        if debug:
            L = np.zeros_like(self.field_seuil)
            L_area = np.zeros_like(self.field_seuil)
            L_sum = np.zeros_like(self.field_seuil)

        sum_field = np.zeros(Nobj)
        sum_normfield = np.zeros(Nobj)

        for i in range(Nobj):
            # print('vs field shape = {}'.format(np.shape(self.field)))
            # print('vs field_seuil shape = {}'.format(np.shape(self.field_seuil)))
            # print('conncomp shape = {}'.format(np.shape(conncomp)))
            to_sum_field = np.zeros_like(self.field_seuil)
            to_sum_normfield = np.zeros_like(self.field_seuil)

            pixels = np.nonzero(conncomp == num_area[i])

            to_sum_field[pixels] = self.field[pixels].copy()
            to_sum_normfield[pixels] = self.normfield[pixels].copy()

            sum_field[i] = np.sum(to_sum_field)
            sum_normfield[i] = np.sum(to_sum_normfield)
            if sum_field[i] < self.seuil:
                print('Proooooooooooooooobleeeeeeeeeeeeeme')
                print('sum = {} vs field {} '.format(sum_normfield[i], self.seuil))

            if debug:
                L[pixels] = i
                L_area[pixels] = Area[i]
                L_sum[pixels] = sum_field[i]

        stop_time = timeit.default_timer()
        # print('tps pour calcul sum :', stop_time - start_time)

        if not fault_analyse:
            conncomp = None

        return Nobj, num_area, Area, center, orientation, sum_field, sum_normfield, np.sum(Area), np.sum(
            sum_field), np.sum(sum_normfield), conncomp


# ---------------------------------------------------------------------------------------------------------------------#
class ImgEvent():
    """
    Classe qui permet e trouver les events dans les champs de déformations.

    Attributes:
        config (class) : config associée à la l'analyse


    """

    # ---------------------------------------------------------#
    def __init__(self, config):
        """
        The constructor for ImgEvent.

        Parameters:
            config (class) : config associée à la l'analyse

        """
        ## Config
        self.config = config

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
    def save_single(self, data, name, extension='npy', nbfichier=None):

        if data is not None:
            if extension == 'npy':
                to_save = self.path_signal + name
                np.save(to_save, data)
            elif extension == 'cell':
                Cell(self.path_signal + name, nbfichier, data=data, extension='cell')
            else:
                Cell(self.path_signal + name, nbfichier, data=data, extension='csv')

    # ------------------------------------------
    def find_field_seuil_k(self, field, seuil):

        field_seuil = np.zeros_like(field)
        for i in range(np.shape(field)[0]):
            for j in range(np.shape(field)[1]):
                a = field[i, j]
                if a > seuil:
                    field_seuil[i, j] = 1

        return field_seuil

    # ------------------------------------------
    def find_field_seuil(self, shape_f, field, seuil, fname, signe=''):

        start_time = timeit.default_timer()
        fileName = 'field_seuil_{}{}_{}s'.format(fname, signe, seuil)
        fileObj = Path(self.path_signal + fileName + '.npy')
        is_fileObj = fileObj.is_file()

        if not is_fileObj:
            field_seuil = np.zeros_like(field)

            for k in range(shape_f.nb_pict):

                for i in range(shape_f.size_w):
                    for j in range(shape_f.size_c):
                        a = field[i, j, k]
                        if a > seuil:
                            field_seuil[i, j, k] = 1
            if fname is not None:
                logging.info("writting into {}".format(fileName))
                self.save_single(field_seuil, fileName)
        else:
            logging.info("loading from {}".format(fileName))
            field_seuil = self.import_single(fileName)
        stop_time = timeit.default_timer()
        print('tps pour seuiler :', stop_time - start_time)

        return field_seuil


# ---------------------------------------------------------------------------------------------------------------------#
class FaultImgEvent(ImgEvent):
    '''  '''

    # ---------------------------------------------------------#
    def __init__(self, config, pxmm, f, normf, posx, posy, posX, posY, seuil, signaltype, fname, NN_data, signe='', num_set=None, already_done=False):
        ImgEvent.__init__(self, config)

        ## Config
        self.config = config
        self.pxmm = pxmm

        self.seuil = seuil

        self.NN_data = NN_data
        self.signaltype, self.fname, self.savename, _, _ = def_names(signaltype, fname, NN_data)

        logging.info("lookint at : {} with seuil = {}".format(fname, seuil))

        if num_set is not None:
            self.fname = '{}_{}'.format(self.fname, num_set)

        logging.info("lookint at : {} with seuil = {}".format(self.fname, self.seuil))

        self.path_signal = self.config.global_path_save + '/pict_event_fault_' + self.signaltype + '/'
        self.to_save_fig = self.config.global_path_save + '/figure_pict_event_fault_' + self.signaltype + '/'

        if f is not None:
            self.shape_f = Shape(f)
            logging.info("field of shape : {}, and norm field of shape {}".format(np.shape(f), np.shape(normf)))
            self.field_seuil = self.find_field_seuil(self.shape_f, f, self.seuil, self.fname, signe)
            self.info = self.regs_analyse(f, normf, self.fname, signe)
            self.info_stats = self.regs_stats(self.fname, signe)
            self.info_reg = self.reg_pos_analyse(signe, posx, posy)
            self.info_reg_XY = self.reg_pos_analyse(signe, posX, posY, '_XY')

        if already_done:
            self.info_stats = self.regs_stats(self.fname, signe)

    # ------------------------------------------
    def regs_analyse(self, f, normf, fname, signe):

        start_time = timeit.default_timer()
        fileName = self.path_signal + 'info_{}{}_{}s.npy'.format(fname, signe, self.seuil)
        logging.info("regs analyse - dealing with file : {}".format(fileName))
        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()

        logging.debug(" shape {} : {}".format(fname, f.shape))

        if not is_fileObj:
            info = dictdata()

            print('seuil des reg analyse = {}'.format(self.seuil))

            for i in range(self.shape_f.nb_pict):
                v = f[:, :, i]
                nv = normf[:, :, i]
                vs = self.field_seuil[:, :, i]
                # if i == self.shape_f.nb_pict - 1:
                #     print('fr i {} v min = {} et v max = {}'.format(i, np.min(v), np.max(v)))

                info_img = InfoField(self.config, v, nv, vs, self.seuil, fname, signe, self.path_signal,
                                     fault_analyse=True)
                info.add('img_{}'.format(i), info_img)
            logging.info("Writing file into : {}".format(fileName))
            np.save(fileName, info)
        else:
            logging.info("Loading fil from : {}".format(fileName))
            info = np.load(fileName, allow_pickle=True).flat[0]

            stop_time = timeit.default_timer()
            print('tps pour seuilreg :', stop_time - start_time)

        return info

    # ------------------------------------------
    def regs_stats(self, fname, signe):
        start_time = timeit.default_timer()
        fileName = self.path_signal + 'info_stats_{}{}_{}s.npy'.format(fname, signe, self.seuil)
        logging.info("reg stats - dealing with file : {}".format(fileName))
        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()
        start_time = timeit.default_timer()

        if not is_fileObj:
            info_stats = dictdata()
            nb_area_tot = 0

            for k in range(self.shape_f.nb_pict):
                nb_area_tot = nb_area_tot + self.info['img_{}'.format(k)].nb_area

            nb_area_img = np.zeros(self.shape_f.nb_pict)
            sum_S_a = np.zeros(self.shape_f.nb_pict)
            sum_S_f = np.zeros(self.shape_f.nb_pict)
            sum_S_nf = np.zeros(self.shape_f.nb_pict)
            center = np.zeros((nb_area_tot, 2))
            orientation = np.zeros(nb_area_tot)
            S_a = np.zeros(nb_area_tot)
            S_f = np.zeros(nb_area_tot)
            S_nf = np.zeros(nb_area_tot)
            pict_S = np.zeros(nb_area_tot)

            ## recup analyse_info par reg
            j = 0
            for k in range(self.shape_f.nb_pict):
                subinfo = self.info['img_{}'.format(k)]

                nb_area_img[k] = subinfo.nb_area

                for i in range(subinfo.nb_area):
                    center[j + i, :] = subinfo.center[i, :]
                    orientation[j + i] = subinfo.orientation[i]
                    S_a[j + i] = subinfo.size_area[i]
                    S_f[j + i] = subinfo.sum_field[i]
                    S_nf[j + i] = subinfo.sum_normfield[i]
                    pict_S[j + i] = k

                    if subinfo.sum_field[i] < self.seuil:
                        print('Proooooooooooooooobleeeeeeeeeeeeeme', i, k)

                j = j + subinfo.nb_area

                sum_S_a[k] = subinfo.size_area_img
                sum_S_f[k] = subinfo.sum_field_img
                sum_S_nf[k] = subinfo.sum_normfield_img

            info_stats.add('nb_area_img', nb_area_img)
            info_stats.add('sum_S_a', sum_S_a)
            info_stats.add('sum_S_f', sum_S_f)
            info_stats.add('sum_S_nf', sum_S_nf)
            info_stats.add('center', center)
            info_stats.add('orientation', orientation)
            info_stats.add('S_a', S_a)
            info_stats.add('S_f', S_f)
            info_stats.add('S_nf', S_nf)
            info_stats.add('pict_S', pict_S)

            logging.info("Writing file into : {}".format(fileName))
            np.save(fileName, info_stats)
            stop_time = timeit.default_timer()
            print('tps pour stat_reg :', stop_time - start_time)
        else:
            logging.info("Loading fil from : {}".format(fileName))
            info_stats = np.load(fileName, allow_pickle=True).flat[0]

        return info_stats

    # ------------------------------------------
    def stichs_analyse(self, field):

        new_field = field[self.field_seuil == 1]

        return new_field

    # ------------------------------------------
    def find_I_J(self, reg):
        ## on cherche les I(row) et J(colone) des pixels appartenant à event i
        # I, J sont les indices dans un array 2D (lw, lc)
        coords = reg.coords
        I = np.array([0])
        J = np.array([0])
        for l in range(coords.shape[0]):
            # print(coords[l], np.array(coords[l][0]), np.array(coords[l][1]))
            I = np.concatenate((I, [coords[l][0]]))
            J = np.concatenate((J, [coords[l][1]]))

        I = I[1::].astype(int)
        J = J[1::].astype(int)

        return I, J

    # ------------------------------------------
    def reconstruct_values_field_from_info(self, field):

        new_field = np.ones_like(field) * np.NAN

        for k in range(self.shape_f.nb_pict):

            # print('nb pict = {} sur {}'.format(k, shape_f.nb_pict))

            subinfo = self.info['img_{}'.format(k)]
            Reg = measure.regionprops(subinfo.conncomp)

            for i in range(subinfo.nb_area):
                # print('area num {}, il y a {} mailles dans cette regions'.format(i, Reg[i].area))

                I, J = self.find_I_J(Reg[i])

                new_field[I, J, k] = field[I, J, k].copy()

        new_shape = Shape(new_field)
        new_field = new_shape.ndim_to_1dim(new_field)

        return new_field[~np.isnan(new_field)]

    # ------------------------------------------
    def pos_on_reg(self, I, J, posx, posy):

        pos_info = dictdata()

        centroid_x = np.mean(posx[I, J])  # moyenne position en x sur base xy
        centroid_y = np.mean(posy[I, J])  # moyenne position en y sur base xy

        ## extremum de distances entre les position des pixel et la moyenne des postions
        max_pos_y = np.max(posy[I, J])
        min_pos_y = np.min(posy[I, J])
        max_pos_x = np.max(posx[I, J])
        min_pos_x = np.min(posx[I, J])

        delta_y = (np.max(posy[I, J]) - np.min(posy[I, J])) / self.pxmm  # taille "reelle" de event projettée sur uy
        delta_x = (np.max(posx[I, J]) - np.min(posx[I, J])) / self.pxmm  # taille "reelle" de event projettée sur uZ

        pix_posy = np.asarray(posy[I, J] - centroid_y)  # position de chaque pixel appartenant à event wrt centre event
        pix_posx = np.asarray(posx[I, J] - centroid_x)  # position de chaque pixel appartenant à event wrt centre event

        # print('la taille de l event en (x, y) est ({}, {}) mm'.format(delta_x, delta_y))

        pos_info.add('centroid_x', centroid_x)
        pos_info.add('centroid_y', centroid_y)
        pos_info.add('min_pos_x', min_pos_x)
        pos_info.add('max_pos_x', max_pos_x)
        pos_info.add('min_pos_y', min_pos_y)
        pos_info.add('max_pos_y', max_pos_y)
        pos_info.add('delta_x', delta_x)
        pos_info.add('delta_y', delta_y)
        pos_info.add('pix_posx', pix_posx)
        pos_info.add('pix_posy', pix_posy)

        return pos_info

    # ------------------------------------------
    def reg_pos_analyse(self, signe, posx, posy, referentiel=''):
        start_time = timeit.default_timer()
        fileName = self.path_signal + 'info_reg{}_{}{}_{}s.npy'.format(referentiel, self.fname, signe, self.seuil)
        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()

        if not is_fileObj:
            info = dictdata()

            print('seuil des reg analyse = {}'.format(self.seuil))

            for k in range(self.shape_f.nb_pict):
                subinfo = self.info['img_{}'.format(k)]
                Reg = measure.regionprops(subinfo.conncomp)
                info_reg = dictdata()

                logging.debug(" shape posx {}".format(posx.shape))

                if (posx.shape[0] % 2 == 0) and (posx.shape[1] % 2 == 0):
                    x = posx[1::, 1::, k]
                    y = posy[1::, 1::, k]
                elif (posx.shape[0] % 2 == 0):
                    x = posx[1::, 1:-1, k]
                    y = posy[1::, 1:-1, k]
                elif (posx.shape[1] % 2 == 0):
                    x = posx[1:-1, 1::, k]
                    y = posy[1:-1, 1::, k]



                logging.debug(" shape posx {}".format(x.shape))

                for i in range(subinfo.nb_area):
                    # print('area num {}, il y a {} mailles dans cette regions'.format(i, Reg[i].area))

                    I, J = self.find_I_J(Reg[i])
                    logging.debug(" info I : {} | J : {}".format(I, J))

                    info_reg_area = self.pos_on_reg(I, J, x, y)
                    info_reg.add('reg_{}'.format(i), info_reg_area)

                info.add('img_{}'.format(k), info_reg)

            logging.info("Writing file into : {}".format(fileName))
            np.save(fileName, info)

        else:
            logging.info("Loading filefrom : {}".format(fileName))
            info = np.load(fileName, allow_pickle=True).flat[0]

        stop_time = timeit.default_timer()
        print('tps pour seuilreg :', stop_time - start_time)

        return info

    # ------------------------------------------
    def M0_mean(self, size, meandu, mu):
        return mu * size * meandu

    # ------------------------------------------
    def Mw(self, M0):
        return 2 / 3 * np.log10(M0) - 6

    # ------------------------------------------
    def analyse_magnitude(self, info_reg, du_field, mu):

        M0_mean = np.array([])
        Mw = np.array([])
        S = np.array([])

        for k in range(self.shape_f.nb_pict):
            subinfo = self.info['img_{}'.format(k)]
            subinfo_reg = info_reg['img_{}'.format(k)]

            Reg = measure.regionprops(subinfo.conncomp)

            for i in range(subinfo.nb_area):
                I, J = self.find_I_J(Reg[i])
                meandu = np.mean(du_field[I, J, k])

                delta_x = subinfo_reg['reg_{}'.format(i)]['delta_x']
                delta_y = subinfo_reg['reg_{}'.format(i)]['delta_y']

                size = np.sqrt(delta_x ** 2 + delta_y ** 2)

                bla = self.M0_mean(size, meandu, mu)
                bli = self.Mw(bla)

                # print(bla, bli)
                M0_mean = np.concatenate((M0_mean, np.array([bla])))
                Mw = np.concatenate((Mw, np.array([bli])))
                S = np.concatenate((S, np.array([size])))

        return M0_mean, Mw, S


# ---------------------------------------------------------------------------------------------------------------------#
class StatsImgEvent(ImgEvent):
    """
    Classe qui permet e trouver les events dans les champs de déformations.

    Attributes:
        config (class) : config associée à la l'analyse

        seuil (int) : valeur utilisée pour seuiller les évents
        save_seuil (str) : matricule du seuil

        signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' nom du dossier
        fname (str) : nom du fichier
        savename (str) : '_' + extension NN
        NN_data (str) : '', 'train', 'val', 'test'
        sep_posneg (bol) : définie la façon de traité les régions positives et négative

        saving step (bol) : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas

        path_signal (str) : chemin du dossier associé à ce signal
        to_save_fig (str) : chemin associé pour save fig

        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse
        cycles (list): liste des cycles dans cette extention NN (not None seulement quand NN et single set)
        sub_cycles_NN (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre total de cycles dans analyse
        NN_sub_cycles (list[liste]) : liste des cycles par set dans ce NN comptés sur nombre de cycles total dans NN

        shape (class) : class shape sur field

        nb_area_img (array) : nombre de regions par img
        S_a_img (1D array) :  tableau des nombre de maille appartenant a regions par img
        S_f_img (1D array) : tableau sum des valeurs de champs sur toutes les regions par img
        S_a (1D array) : taille en nombre de mailles de chaque region
        S_f (1D array) : sum des valeurs du champs sur les mailles d'une region, pour toutes les regions
        pict_S ( 1D array) : labeled array, chaque region d'une même img est labélisée par le numéro de l'img

    """

    # ---------------------------------------------------------#
    def __init__(self, config, f, seuil, save_seuil, signaltype, NN_data, fname, sep_posneg, saving_step=True):
        """
        The constructor for ImgEvent.

        Parameters:
            config (class) : config associée à la l'analyse

            f (1D array) : field
            seuil (int) : valeur utilisée pour seuiller les évents
            save_seuil (str) : matricule du seuil

            signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force
            NN_data (str) : '', 'train', 'val', 'test' extension NN
            fname (str) : nom du champ
            sep_posneg (bol) : définie la façon de traité les régions positives et négative

            saving step (bol) : pêrmet de sauver
        """
        ImgEvent.__init__(self, config)

        ## Config

        self.seuil = seuil
        self.save_seuil = save_seuil

        self.NN_data = NN_data
        self.sep_posneg = sep_posneg
        self.signaltype, self.fname, self.savename, _, _ = def_names(signaltype, fname, NN_data)

        self.saving_step = saving_step

        self.path_signal = self.config.global_path_save + '/pict_event_' + self.signaltype + '/'
        self.to_save_fig = self.config.global_path_save + '/figure_pict_event_' + self.signaltype + '/'

        self.nbcycle, self.cycles, self.sub_cycles, self.sub_cycles_NN, self.NN_sub_cycles = def_nbcycle(self.config,
                                                                                                            self.path_signal,
                                                                                                            self.fname,
                                                                                                            self.NN_data)


        self.shape_f = Shape(f)

        ## regarde si le fichier existe dejà :
        if fname == 'vort' and self.sep_posneg:
            fileName = self.path_signal + 'S_f_' + self.fname + '_p_' + self.save_seuil + '.npy'
        else:
            fileName = self.path_signal + 'S_f_' + self.fname + '_' + self.save_seuil + '.npy'
        fileObj = Path(fileName)
        is_fileObj = fileObj.is_file()

        if is_fileObj:
            print('seuil déjà enregisté')
            if fname == 'vort' and self.sep_posneg:
                self.nb_area_img_p, self.nb_area_img_n, self.sum_S_a_p, self.sum_S_a_n, self.S_a_p, self.S_a_n, \
                self.sum_S_f_p, self.sum_S_f_n, self.S_f_p, self.S_f_n, \
                self.pict_S_p, self.pict_S_n = self.import_data(fname)
            else:
                self.nb_area_img, self.sum_S_a, self.S_a, self.sum_S_f, self.S_f, \
                self.pict_S = self.import_data(fname)
        else:
            print('seuil à traiter')
            info, info_p, info_n = self.reg_analyse(fname, f, self.shape_f, self.seuil)

            if fname == 'vort' and self.sep_posneg:
                self.nb_area_img_p, self.sum_S_a_p, self.sum_S_f_p, self.S_a_p, self.S_f_p, \
                self.pict_S_p = self.stat_reg(info=info_p)
                self.nb_area_img_n, self.sum_S_a_n, self.sum_S_f_n, self.S_a_n, self.S_f_n, \
                self.pict_S_n = self.stat_reg(info=info_n)
            else:
                self.nb_area_img, self.sum_S_a, self.sum_S_f, self.S_a, self.S_f, \
                self.pict_S = self.stat_reg(info=info)

            ## save
            if self.saving_step:
                if fname == 'vort' and self.sep_posneg:
                    self.save_single(self.nb_area_img_p, 'nb_area_img_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.sum_S_a_p, 'sum_S_a_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.S_a_p, 'S_a_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.sum_S_f_p, 'sum_S_f_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.S_f_p, 'S_f_' + self.fname + '_p_' + self.save_seuil)
                    self.save_single(self.pict_S_p, 'pict_S_' + self.fname + '_p_' + self.save_seuil)

                    self.save_single(self.nb_area_img_n, 'nb_area_img_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.sum_S_a_n, 'sum_S_a_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.S_a_n, 'S_a_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.sum_S_f_n, 'sum_S_f_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.S_f_n, 'S_f_' + self.fname + '_n_' + self.save_seuil)
                    self.save_single(self.pict_S_n, 'pict_S_' + self.fname + '_n_' + self.save_seuil)

                else:
                    self.save_single(self.nb_area_img, 'nb_area_img_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.sum_S_a, 'sum_S_a_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.S_a, 'S_a_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.sum_S_f, 'sum_S_f_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.S_f, 'S_f_' + self.fname + '_' + self.save_seuil)
                    self.save_single(self.pict_S, 'pict_S_' + self.fname + '_' + self.save_seuil)

    # ------------------------------------------
    def import_data(self, fname):
        if fname == 'vort' and self.sep_posneg:
            nb_area_img_p = self.import_single('nb_area_img_' + self.fname + '_p_' + self.save_seuil)
            sum_S_a_p = self.import_single('sum_S_a_' + self.fname + '_p_' + self.save_seuil)
            S_a_p = self.import_single('S_a_' + self.fname + '_p_' + self.save_seuil)
            sum_S_f_p = self.import_single('sum_S_f_' + self.fname + '_p_' + self.save_seuil)
            S_f_p = self.import_single('S_f_' + self.fname + '_p_' + self.save_seuil)
            pict_S_p = self.import_single('pict_S_' + self.fname + '_p_' + self.save_seuil)

            nb_area_img_n = self.import_single('nb_area_img_' + self.fname + '_n_' + self.save_seuil)
            sum_S_a_n = self.import_single('sum_S_a_' + self.fname + '_n_' + self.save_seuil)
            S_a_n = self.import_single('S_a_' + self.fname + '_n_' + self.save_seuil)
            sum_S_f_n = self.import_single('sum_S_f_' + self.fname + '_n_' + self.save_seuil)
            S_f_n = self.import_single('S_f_' + self.fname + '_n_' + self.save_seuil)
            pict_S_n = self.import_single('pict_S_' + self.fname + '_n_' + self.save_seuil)

            return nb_area_img_p, nb_area_img_n, sum_S_a_p, sum_S_a_n, S_a_p, S_a_n, sum_S_f_p, sum_S_f_n, S_f_p, S_f_n, \
                   pict_S_p, pict_S_n

        else:
            nb_area_img = self.import_single('nb_area_img_' + self.fname + '_' + self.save_seuil)
            sum_S_a = self.import_single('sum_S_a_' + self.fname + '_' + self.save_seuil)
            S_a = self.import_single('S_a_' + self.fname + '_' + self.save_seuil)
            sum_S_f = self.import_single('sum_S_f_' + self.fname + '_' + self.save_seuil)
            S_f = self.import_single('S_f_' + self.fname + '_' + self.save_seuil)
            pict_S = self.import_single('pict_S_' + self.fname + '_' + self.save_seuil)

            return nb_area_img, sum_S_a, S_a, sum_S_f, S_f, pict_S

    # ------------------------------------------
    def reg_analyse(self, fname, f, shape_f, seuil):

        info = [0 for i in range(shape_f.nb_pict)]
        info_p = [0 for i in range(shape_f.nb_pict)]
        info_n = [0 for i in range(shape_f.nb_pict)]

        if fname == 'vort' and self.sep_posneg:
            field_seuil_p = self.find_field_seuil(f, seuil, fname, '_p')
            field_seuil_n = self.find_field_seuil(-f, seuil, fname, '_n')
        elif fname == 'vort' and not self.sep_posneg:
            field_seuil = self.find_field_seuil(np.abs(f), seuil, fname)
        else:
            field_seuil = self.find_field_seuil(f, seuil, fname)

        start_time = timeit.default_timer()

        for k in range(shape_f.nb_pict):
            v = f[:, :, k]

            if fname == 'vort' and self.sep_posneg:
                vs_p = field_seuil_p[:, :, k]
                vs_n = field_seuil_n[:, :, k]

                info_p[k] = InfoField(self.config, v, vs_p, seuil)
                info_n[k] = InfoField(self.config, -v, vs_n, seuil)
                info[k] = None
            elif fname == 'vort' and not self.sep_posneg:
                vs = field_seuil[:, :, k]

                info[k] = InfoField(self.config, np.abs(v), vs, seuil)
                info_p[k] = None
                info_n[k] = None
            else:
                vs = field_seuil[:, :, k]

                info[k] = InfoField(self.config, v, vs, seuil)
                info_p[k] = None
                info_n[k] = None

        stop_time = timeit.default_timer()
        print('tps pour seuilreg :', stop_time - start_time)

        return info, info_p, info_n

    # ------------------------------------------
    def stat_reg(self, info):

        start_time = timeit.default_timer()

        nb_area_tot = 0

        for k in range(self.shape_f.nb_pict):
            nb_area_tot = nb_area_tot + info[k].nb_area

        nb_area_img = np.zeros(self.shape_f.nb_pict)
        sum_S_a = np.zeros(self.shape_f.nb_pict)
        sum_S_f = np.zeros(self.shape_f.nb_pict)
        S_a = np.zeros(nb_area_tot)
        S_f = np.zeros(nb_area_tot)
        pict_S = np.zeros(nb_area_tot)

        ## recup analyse_info par reg
        j = 0
        for k in range(self.shape_f.nb_pict):
            subinfo = info[k]

            nb_area_img[k] = subinfo.nb_area

            for i in range(subinfo.nb_area):
                S_a[j + i] = subinfo.size_area[i]
                S_f[j + i] = subinfo.sum_field[i]
                pict_S[j + i] = k

                if subinfo.sum_field[i] < self.seuil:
                    print('Proooooooooooooooobleeeeeeeeeeeeeme', i, k)

            j = j + subinfo.nb_area

            sum_S_a[k] = subinfo.size_area_img
            sum_S_f[k] = subinfo.sum_field_img

        stop_time = timeit.default_timer()
        print('tps pour stat_reg :', stop_time - start_time)

        return nb_area_img, sum_S_a, sum_S_f, S_a, S_f, pict_S