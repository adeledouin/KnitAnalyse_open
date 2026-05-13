import numpy as np
import math

class Config():
    ''' def paramètres de la manip
    et paramètres de l'analyse '''

    # ---------------------------------------------------------#
    def __init__(self, path_from_root, dictionnaire_param_exp):
        '''param ref : définit le set auquel appartient la manip
        param mix : est ce que mix set ?
        param num_set : nombre de set
         param date : date de la manip
         param nexp : numero de test de prestat
         param version : version du code
         param globa_path : chemin global pour atteindre le fichier d'analyse
         param v : vitesse de traction en mm/s
         param Lw_0 : extension initiale
         param Lw_max : extension maximale
         param fr : frequence d'acquisition de l'instron
         param prescycle : nombre de pres cycles
         param mincycle : numero du premier cycle à garder
         param maxcycle : numero du dernier cycle à garder
         param nbcycle : nombre de cycles à garder
         '''

        self.dict_exp = dictionnaire_param_exp

        self.mix_set = self.dict_exp['mix_set']
        self.nb_set = self.dict_exp['nb_set']
        self.img = self.dict_exp['img']

        self.ref = self.dict_exp['ref']
        self.date = self.dict_exp['date']
        self.nexp = self.dict_exp['nexp']
        self.version_raw = self.dict_exp['version_raw']
        self.version_work = self.dict_exp['version_work']

        self.v = self.dict_exp['vitesse']
        self.Lw_0 = self.dict_exp['Lw_0']
        self.Lw_i = self.dict_exp['Lw_i']
        self.Lw_max = self.dict_exp['Lw_max']
        self.fr = self.dict_exp['fr_instron']  #Hz => dt = 0.04s
        self.sursample = self.dict_exp['sursample']
        self.reso_instron = 0.00015  # mm
        self.imgred = self.dict_exp['red']

        self.prescycle = self.dict_exp['prescycle']
        self.mincycle = self.dict_exp['mincycle']
        self.maxcycle = self.dict_exp['maxcycle']
        self.nbcycle = np.asarray(self.maxcycle) - np.asarray(self.mincycle) + 1

        if self.mix_set:
            self.sub_cycles = [0 for i in range(self.nb_set)]
            k = 0
            cycles_tot = np.arange(np.sum(self.nbcycle))
            for i in range(self.nb_set):
                self.sub_cycles[i] = cycles_tot[k: k + self.nbcycle[i]]
                k = k + self.nbcycle[i]

        self.nb_process = 15

        if not self.mix_set:
            self.global_path_load_raw = '/{}/KnitAnalyse/'.format(path_from_root) + self.ref + \
                                        '/%d/exp%d/version%d/' % (self.date, self.nexp, self.version_raw)
            self.global_path_load = '/{}/KnitAnalyse/'.format(path_from_root) + self.ref + \
                                    '/%d/exp%d/version%d/' % (self.date, self.nexp, self.version_work)
            self.global_path_save = '/{}/KnitAnalyse/'.format(path_from_root) + self.ref + \
                                    '/%d/exp%d/version%d/' % (self.date, self.nexp, self.version_work)
        else:
            self.global_path_load_raw = '/{}/KnitAnalyse/'.format(path_from_root) + self.ref + \
                                    '/%d/exp%d/version%d/'
            self.global_path_load = '/{}/KnitAnalyse/'.format(path_from_root) + self.ref + \
                                    '/mix' + '/version%d/' % (self.version_work)
            self.global_path_save = '/{}/KnitAnalyse/'.format(path_from_root) + self.ref + \
                                    '/mix/version%d/' % (self.version_work)

    # ------------------------------------------
    def config_prepro(self):
        '''config pour class prepro
        param Lw_1 : extension min à garder pour algofit
        param Lw_2 : extension max à garder pour algofit '''

        Lw_1 = self.Lw_i - self.Lw_0
        Lw_2 = self.Lw_max - self.Lw_0
        force_ref = 0.5

        return Lw_1, Lw_2, force_ref

    # ------------------------------------------
    def config_prepro_img(self):
        '''config pour class prepro quand img '''

        delta_t_pict = self.dict_exp['delta_t_pict']  # s
        delta_z = self.dict_exp['delta_z']
        first_trigger = 1  # s

        return delta_z, delta_t_pict, first_trigger

    # ------------------------------------------
    def config_flu(self):
        '''config pour class prepro
        param Lw_1 : extension min à garder pour algofit '''

        dext_instron = 2*self.reso_instron
        # print(dext_instron, self.v)
        dt_instron = dext_instron/self.v/2
        nb_point_instron = math.ceil(dt_instron*self.fr)
        nb_point_eff = self.dict_exp['nb_point_eff']
        Lw_1 = 1 #mm

        if self.sursample:
            print('___________nb point instron___________', nb_point_instron)
            print('___________nb point supp____________', nb_point_instron*2+1)
            print('___________nb point eff___________', nb_point_eff*2+1)

        return Lw_1, nb_point_eff

    # ------------------------------------------
    def config_corr(self):

        fraction = self.dict_exp['config_corr_fraction']

        return fraction

    # ------------------------------------------
    def config_scalarevent(self, Sm):
        '''config pour class scalarevents_brut
        param exposants : exposants ddes seuils utilisés dans le plot pdf des events
        param seuils : seuil utilisés dans le plot pdf events
        param nb_seuils : nb de seuls testé dans le plot pdf events
        param wich_seuil : index du seuil à utilisé pour la stat des evenements
        param : nbclasses : nombre de classes quand plot les stats des events'''

        exposants = self.dict_exp['config_scalarevent_flu_exposants']
        seuils = self.dict_exp['config_scalarevent_flu_seuils']
        save_seuils = self.dict_exp['config_scalarevent_flu_save_seuils']
        nb_seuils = self.dict_exp['config_scalarevent_flu_nb_seuils']
        if Sm:
            which_seuil = self.dict_exp['config_scalarevent_flu_Sm_which_seuil']
        else:
            which_seuil = self.dict_exp['config_scalarevent_flu_which_seuil']
        nbclasses = self.dict_exp['config_scalarevent_flu_nbclasses']

        return exposants, seuils, save_seuils, nb_seuils, which_seuil, nbclasses

    # ------------------------------------------
    def config_imgevent(self, signaltype, fsave):

        if fsave == '_slip_X' or fsave == '_slip_Y':
            exposants = self.dict_exp['config_imgevent_flu_slip_exposants']
            seuils = self.dict_exp['config_imgevent_flu_slip_seuils']
            save_seuils = self.dict_exp['config_imgevent_flu_slip_save_seuils']
            nb_seuils = self.dict_exp['config_imgevent_flu_slip_nb_seuils']
            which_seuil = self.dict_exp['config_imgevent_flu_slip_which_seuil']
        elif fsave == '_vort' or fsave == '_vort_pn' or fsave == '_dev':
            exposants = self.dict_exp['config_imgevent_flu_sepvort_exposants']
            seuils = self.dict_exp['config_imgevent_flu_sepvort_seuils']
            save_seuils = self.dict_exp['config_imgevent_flu_sepvort_save_seuils']
            nb_seuils = self.dict_exp['config_imgevent_flu_sepvort_nb_seuils']
            which_seuil = self.dict_exp['config_imgevent_flu_sepvort_which_seuil']
        else:
            exposants = self.dict_exp['config_imgevent_flu_exposants']
            seuils = self.dict_exp['config_imgevent_flu_seuils']
            save_seuils = self.dict_exp['config_imgevent_flu_save_seuils']
            nb_seuils = self.dict_exp['config_imgevent_flu_nb_seuils']
            which_seuil = self.dict_exp['config_imgevent_flu_which_seuil']

        return exposants, seuils, which_seuil, nb_seuils, save_seuils

    # ------------------------------------------
    def config_knitquakes(self):

        pxmm = 4938/220 ## self.dict_exp['config_knitquakes_pixel']/self.dict_exp['config_knitquakes_Lw']
        exposant = 2
        seuil = 3e-2
        save_seuil = '3_2'

        return pxmm, exposant, seuil, save_seuil