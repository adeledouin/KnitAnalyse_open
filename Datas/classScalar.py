from functools import partial
import scipy.io as spo
from multiprocessing import Pool, freeze_support
import timeit
import scipy.optimize as opt
import numpy as np

from Utils.classFindPeak import FindPeak
from Utils.classCell import Cell

# ---------------------------------------------------------------------------------------------------------------------#
class Preprocess():
    ''' à partir des données brutes de l'Instron récupère froce, extension et temps, associés à zone d'interêt  '''

    # ---------------------------------------------------------#
    def __init__(self, config, remote, display_figure=True, saving_step=True):
        ''' display_figure : affiche les figures pour vérifier pas de pb dans l'analyse
            saving step : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas'''

        ## Config
        self.config = config
        self.remote = remote

        self.nb_processes = config.nb_process
        self.display_figure = display_figure
        self.saving_step = saving_step

        self.lw1, self.lw2, self.force_ref = self.config.config_prepro()

        self.to_save_brut = self.config.global_path_save + 'm_f/'
        self.to_save_fig = self.config.global_path_save + 'figure_m_f/'

        ## obj créés
        if self.saving_step:
            self.tosave_segmforce = self.to_save_brut + 'segmforce'
            self.tosave_segmtime = self.to_save_brut + 'segmtime'
            self.tosave_segmext = self.to_save_brut + 'segmext'

            self.tosave_indice_lw1 = self.to_save_brut + 'indice_lw1'
            self.tosave_indice_lw2 = self.to_save_brut + 'indice_lw2'
            self.tosave_Lw_f_ref = self.to_save_brut + 'Lw_f_ref'
            self.tosave_f_ref = self.to_save_brut + 'f_ref'

            self.tosave_mtt_f = self.to_save_brut + 'mtt_f'
            self.tosave_mtt_t = self.to_save_brut + 'mtt_t'
            self.tosave_mtt_trenorm = self.to_save_brut + 'mtt_trenorm'
            self.tosave_mtt_ext = self.to_save_brut + 'mtt_ext'
            self.tosave_mtt_size = self.to_save_brut + 'mtt_size'

            self.tosave_m1_f = self.to_save_brut + 'm1_f'
            self.tosave_m1_t = self.to_save_brut + 'm1_t'
            self.tosave_m1_trenorm = self.to_save_brut + 'm1_trenorm'
            self.tosave_m1_ext = self.to_save_brut + 'm1_ext'
            self.tosave_m1_size = self.to_save_brut + 'm1_size'

            self.tosave_m3_f = self.to_save_brut + 'm3_f'
            self.tosave_m3_t = self.to_save_brut + 'm3_t'
            self.tosave_m3_trenorm = self.to_save_brut + 'm3_trenorm'
            self.tosave_m3_ext = self.to_save_brut + 'm3_ext'
            self.tosave_m3_size = self.to_save_brut + 'm3_size'

            self.tosave_m_f = self.to_save_brut + 'm_f'
            self.tosave_m_t = self.to_save_brut + 'm_t'
            self.tosave_m_trenorm = self.to_save_brut + 'm_trenorm'
            self.tosave_m_ext = self.to_save_brut + 'm_ext'
            self.tosave_m_size = self.to_save_brut + 'm_size'

            self.tosave_index_picture_m = self.to_save_brut + 'index_picture_m'
            self.tosave_number_picture_m = self.to_save_brut + 'number_picture_m'

            self.tosave_nb_index_picture_m = self.to_save_brut + 'nb_index_picture_m'

    # ------------------------------------------
    def import_mix(self, toload, num_set):

        maxcycle = self.config.maxcycle[num_set]

        indice_lw1 = np.load(toload + 'indice_lw1' + '.npy')
        Lw_f_ref = np.load(toload + 'Lw_f_ref' + '.npy')
        f_ref = np.load(toload + 'f_ref' + '.npy')
        m_size = np.load(toload + 'm_size' + '.npy')

        recup_m_f = Cell(toload + 'm_f', maxcycle)
        recup_m_t = Cell(toload + 'm_trenorm', maxcycle)
        recup_m_ext = Cell(toload + 'm_ext', maxcycle)
        if self.config.img:
            recup_index_picture_m = Cell(toload + 'index_picture_m', maxcycle)
            recup_number_picture_m = Cell(toload + 'number_picture_m', maxcycle)

        m_f = recup_m_f.reco_cell()
        m_t = recup_m_t.reco_cell()
        m_ext = recup_m_ext.reco_cell()
        if self.config.img:
            index_picture_m = recup_index_picture_m.reco_cell()
            number_picture_m = recup_number_picture_m.reco_cell()
        else:
            index_picture_m = None
            number_picture_m = None

        return indice_lw1, Lw_f_ref, f_ref, m_f, m_t, m_ext, m_size, index_picture_m, \
               number_picture_m

    # ------------------------------------------
    def mat_to_py(self, path, name):

        data = spo.loadmat(path)
        data = data[name]

        return data

    # ------------------------------------------
    def synchro_pict(self, n, time, time_picture_norm, N_pict_i):

        A = np.min(np.abs(time - time_picture_norm[N_pict_i + n - 1]))
        I_ext = np.where(np.abs(time - time_picture_norm[N_pict_i + n - 1]) == A)[0][0]

        return I_ext

    # ------------------------------------------
    def main_run(self, plot, data, picturetimeraw, triggertimeraw, num_set=None):
        ''' étape 1 : segmente signal instron en cycle
        étape 2 : extrait uniquement la montée pour tous les cycles
        étape 3 : extrait la partie du signal à vitesse 'lente '''

        ## config
        if not self.config.mix_set:
            maxcycle = self.config.maxcycle
            prescycle = self.config.prescycle
        else:
            maxcycle = self.config.maxcycle[num_set]
            prescycle = self.config.prescycle[num_set]

        if self.config.img:
            delta_z, delta_t_pict, first_trigger = self.config.config_prepro_img()

        ## import
        time = np.asarray(data[:, 0])
        extension = np.asarray(data[:, 1])
        force = np.asarray(data[:, 2])

        data_size = np.size(time)

        if self.config.img:
            nb_picture = picturetimeraw[:, 1]
            time_picture = picturetimeraw[:, 0]
            nb_trigger = triggertimeraw[:, 1]
            time_trigger = triggertimeraw[:, 0]

            ## pict initiale
            time_picture_norm = (time_picture - time_trigger[0]) / 1000 + first_trigger

            N_pict_i = 1
            N_pict_f = np.size(nb_picture)

            Num_pict = int(N_pict_f - N_pict_i + 1)

            print('je retrouve les index')
            start_time = timeit.default_timer()
            # print(__name__)
            if __name__ == "Datas.classScalar":
                with Pool(processes=self.nb_processes) as pool:
                    freeze_support()
                    I_ext = pool.map(partial(self.synchro_pict, time=time, time_picture_norm=time_picture_norm,
                                             N_pict_i=N_pict_i), range(Num_pict))

            index_picture = np.sort(I_ext)

            if delta_z != 1:
                to_keep = self.mat_to_py(self.config.global_path_load_raw + 'pict/img_to_keep', 'img_to_keep')
                # to_keep = np.arange(0, np.size(index_picture), delta_z)
                index_picture = index_picture[to_keep-1]
                N_pict_i = 1
                N_pict_f = np.size(index_picture)

                Num_pict = int(N_pict_f - N_pict_i + 1)


            index_picture_initial = np.zeros(data_size)
            number_picture_initial = np.ones(data_size) * np.nan

            index_picture_initial[index_picture] = 1
            number_picture_initial[index_picture] = np.arange(0, Num_pict).astype(int)

        if self.display_figure:
            plot.plot_x_y(time, force, 't (s)', 'F (N)', 'time', 'force',
                               save=self.to_save_fig, title='force vs time', label=None, grid=None, pts='-')

            plot.plot_x_y(extension, force, 'L_{w} (mm)', 'F (N)', 'ext', 'force',
                               save=self.to_save_fig, title='force vs ext', label=None, grid=None, pts='-')

            plot.plot_x_y(time, extension, 't (s)', 'L_{w} (mm)', 'time', 'ext',
                               save=self.to_save_fig, title='ext vs time', label=None, grid=None, pts='-')

        ## segmantation en cycles
        signal_brut = np.asarray(np.diff(np.sign(extension)))

        findpeak = FindPeak(signal_brut, 1, brut_signal=True)

        ext_zeroindice, _, ext_zeroindice_size, _ = findpeak.recup_min_max_indices()

        if self.display_figure:
            fig, ax = plot.belleFigure('$t(s)$', '$L_{w} (mm)$', nfigure=None)
            ax.plot(time, extension, 'b')
            ax.plot(time[ext_zeroindice], extension[ext_zeroindice], 'rx')
            save = self.to_save_fig + 'extention' + '_vs_' + 'time'
            plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save)

        c_t = [0 for i in range(maxcycle)]
        c_ext = [0 for i in range(maxcycle)]
        c_f = [0 for i in range(maxcycle)]

        if self.config.img:
            index_picture_c = [0 for i in range(maxcycle)]
            number_picture_c = [0 for i in range(maxcycle)]

        print('je segmente en cycle')
        for i in range(maxcycle):
            indexup1 = ext_zeroindice[i + prescycle]
            if i == maxcycle - 1:
                c_t[i] = time[indexup1::]
                c_ext[i] = extension[indexup1::]
                c_f[i] = force[indexup1::]
                if self.config.img:
                    index_picture_c = index_picture_initial[indexup1::]
                    number_picture_c = number_picture_initial[indexup1::]
            else:
                indexup2 = ext_zeroindice[i + 1 + prescycle]
                c_t[i] = time[indexup1:indexup2]
                c_ext[i] = extension[indexup1:indexup2]
                c_f[i] = force[indexup1:indexup2]
                if self.config.img:
                    index_picture_c[i] = index_picture_initial[indexup1:indexup2]
                    number_picture_c[i] = number_picture_initial[indexup1:indexup2]

        if self.display_figure:
            fig, ax = plot.belleFigure('$L_{w} (mm)$', '$F(N)$', nfigure=None)
            ax.plot(self.config.Lw_0 + c_ext[1], c_f[1] - c_f[1][0])
            save = self.to_save_fig + 'F' + '_vs_' + 'L_w_c1'
            plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save)

            fig, ax = plot.belleFigure('$t(s)$', '$F(N)$', nfigure=None)
            for i in range(maxcycle):
                ax.plot(c_t[i] - c_t[i][0], c_f[i])
            save = self.to_save_fig + 'F' + '_vs_' + 't'
            plot.fioritures(ax, fig, title='f vs t pour touts les cycles', label=None, grid=None, save=save)

        ## caractérisation du drift
        Lw_f_ref = np.zeros(maxcycle)
        f_ref = np.zeros(maxcycle)
        for i in range(maxcycle):
            where = np.where(c_f[i] - c_f[i][0] > self.force_ref)[0][0]
            Lw_f_ref[i] = self.config.Lw_0 + c_ext[i][where]
            f_ref[i] = c_f[i][where] - c_f[i][0]

        if self.display_figure:
            # Define the colors to be used using rainbow map (or any other map)
            colors = plot.make_colors(maxcycle)

            plot.plot_x_y_multiliste(np.size(Lw_f_ref), Lw_f_ref, f_ref, 'L_{w} (mm)', 'F(N)', 'Lw_drift', 'F',
                                    save=self.to_save_fig,
                                    title='caracterisation  drift : force à extension de changement de regime',
                                    colors=colors)

            plot.plot_y(Lw_f_ref, 'cycle', 'L_{w} (mm)', 'cycles', 'Lw_drift',
                       save=self.to_save_fig,
                       title='caracterisation  drift : extension de changement de regime',
                       grid=None)

        ## shift affine : continuité en force début et fin de cycle

        f = [0 for i in range(maxcycle)]
        t_renorm = [0 for i in range(maxcycle)]
        droite = [0 for i in range(maxcycle)]

        for i in range(maxcycle):
            t_renorm[i] = c_t[i] - c_t[i][1]
            a = (c_f[i][-1] - c_f[i][0]) / (t_renorm[i][-1] - t_renorm[i][0])
            droite[i] = a * t_renorm[i] + c_f[i][1]
            f[i] = c_f[i] - droite[i]

        ## ségmentation montée

        signal_brut = np.asarray(np.diff(np.sign(extension - (self.config.Lw_max - self.config.Lw_0))))
        findpeak = FindPeak(signal_brut, 1, brut_signal=True)

        _, ext_dixindice, _, ext_dixindice_size = findpeak.recup_min_max_indices()

        if self.display_figure:
            fig, ax = plot.belleFigure('$t(s)$', '$L_{w} (mm)$', nfigure=None)
            ax.plot(time, extension, 'b')
            ax.plot(time[ext_dixindice], extension[ext_dixindice], 'rx')
            save = self.to_save_fig + 'extention' + '_vs_' + 'time'
            plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save)

        mtt_t = [0 for i in range(maxcycle)]
        mtt_trenorm = [0 for i in range(maxcycle)]
        mtt_ext = [0 for i in range(maxcycle)]
        mtt_f = [0 for i in range(maxcycle)]
        mtt_size = np.zeros(maxcycle, dtype=int)
        if self.config.img:
            index_picture_mtt = [0 for i in range(maxcycle)]
            number_picture_mtt = [0 for i in range(maxcycle)]

        print('je recup la montée')
        for i in range(maxcycle):
            indexup1 = ext_zeroindice[i + prescycle]
            indexup2 = ext_dixindice[i + prescycle]
            mtt_t[i] = time[indexup1:indexup2 + 1]
            mtt_trenorm[i] = mtt_t[i] - mtt_t[i][0]
            mtt_ext[i] = extension[indexup1:indexup2 + 1]
            mtt_f[i] = force[indexup1:indexup2 + 1]
            mtt_size[i] = mtt_t[i].size
            if self.config.img:
                index_picture_mtt[i] = index_picture_initial[indexup1:indexup2 + 1]
                number_picture_mtt[i] = number_picture_initial[indexup1:indexup2 + 1]

        if self.config.img:
            nb_index_picture_mtt = 0

            for i in range(maxcycle):
                nb_index_picture_mtt = nb_index_picture_mtt + np.sum(index_picture_mtt[i])

        if self.display_figure:
            plot.plot_x_y_multiliste(maxcycle, mtt_ext, mtt_f, 'L_{w} (mm)', 'F(N)', 'mtt_ext', 'mtt_f',
                                          save=self.to_save_fig,
                                          title='f vs L_w pour toutes les montées', pts='-')

            fig, ax = plot.belleFigure('$L_{w} (mm)$', '$F(N)$', nfigure=None)
            for i in range(maxcycle):
                ax.plot(self.config.Lw_0 + mtt_ext[i], mtt_f[i], )
            save = self.to_save_fig + 'mtt_f' + '_vs_' + 'Lw'
            plot.fioritures(ax, fig, title='f vs L_w pour toutes les montées', label=None, grid=None, save=save)

            plot.plot_x_y_multiliste(maxcycle, mtt_trenorm, mtt_f, 't (s)', 'F(N)', 'mtt_t', 'mtt_f',
                                          save=self.to_save_fig,
                                          title='f vs t pour toutes les montées', pts='-')

            if self.config.img:
                fig, ax = plot.belleFigure('$L_{w} (mm)$ of pict', '$F(N)$ of pict', nfigure=None)
                where_pict = np.where(index_picture_mtt[1] == 1)[0]
                ax.plot(self.config.Lw_0 + mtt_ext[1][where_pict], mtt_f[1][where_pict], '.')
                save = self.to_save_fig + 'pict_ds_mtt_c1'
                plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save)

                fig, ax = plot.belleFigure('$L_{w} (mm)$', '$F(N)$', nfigure=None)
                for i in range(maxcycle):
                    where_pict = np.where(index_picture_mtt[i] == 1)[0]
                    ax.plot(self.config.Lw_0 + mtt_ext[i][where_pict], mtt_f[i][where_pict], '.')
                save = self.to_save_fig + 'pict_ds_mtt'
                plot.fioritures(ax, fig, title='picture dans mtt', label=None, grid=None, save=save)

        ## ségmentation en différents régimes
        print('extrait m')
        indice_lw1 = np.zeros(maxcycle)

        # partie 1
        m1_t = [0 for i in range(maxcycle)]
        m1_trenorm = [0 for i in range(maxcycle)]
        m1_ext = [0 for i in range(maxcycle)]
        m1_f = [0 for i in range(maxcycle)]
        m1_size = np.zeros(maxcycle, dtype=int)

        if self.config.img:
            index_picture_m1 = [0 for i in range(maxcycle)]
            number_picture_m1 = [0 for i in range(maxcycle)]

        for i in range(maxcycle):
            j = np.where(mtt_ext[i] <= self.lw1)[0][-1]
            indice_lw1[i] = j
            m1_t[i] = mtt_t[i][0:j]
            m1_trenorm[i] = mtt_trenorm[i][0:j]
            m1_ext[i] = mtt_ext[i][0:j]
            m1_f[i] = mtt_f[i][0:j]
            m1_size[i] = m1_t[i].size
            if self.config.img:
                index_picture_m1[i] = index_picture_mtt[i][0:j]
                number_picture_m1[i] = number_picture_mtt[i][0:j]

        if self.config.img:
            nb_index_picture_m1 = 0

            for i in range(maxcycle):
                nb_index_picture_m = nb_index_picture_m1 + np.sum(index_picture_m1[i])

        ## partie 2
        m_t = [0 for i in range(maxcycle)]
        m_trenorm = [0 for i in range(maxcycle)]
        m_ext = [0 for i in range(maxcycle)]
        m_f = [0 for i in range(maxcycle)]
        m_size = np.zeros(maxcycle, dtype=int)

        index_picture_m = [0 for i in range(maxcycle)]
        number_picture_m = [0 for i in range(maxcycle)]

        for i in range(maxcycle):
            a = int(indice_lw1[i])
            m_t[i] = mtt_t[i][a::]
            m_trenorm[i] = mtt_trenorm[i][a::]
            m_ext[i] = mtt_ext[i][a::]
            m_f[i] = mtt_f[i][a::]
            m_size[i] = m_t[i].size
            if self.config.img:
                index_picture_m[i] = index_picture_mtt[i][a::]
                number_picture_m[i] = number_picture_mtt[i][a::]

        nb_index_picture_m = 0

        if self.config.img:
            for i in range(maxcycle):
                nb_index_picture_m = nb_index_picture_m + np.sum(index_picture_m[i])

        if self.display_figure:
            fig, ax = plot.belleFigure('$L_{w} (mm)$', '$F(N)$', nfigure=None)
            for i in range(maxcycle):
                ax.plot(self.config.Lw_0 + m_ext[i], m_f[i], )
            save = self.to_save_fig + 'm_f' + '_vs_' + 'Lw'
            plot.fioritures(ax, fig, title='f vs L_w pour toutes les montées', label=None, grid=None, save=save)

            plot.plot_x_y_multiliste(maxcycle, mtt_trenorm, mtt_f, 't (s)', 'F(N)', 'mtt_t', 'mtt_f',
                                          save=self.to_save_fig,
                                          title='f vs t pour toutes les montées', pts='-')

            if self.config.img:
                fig, ax = plot.belleFigure('$L_{w} (mm)$ of pict', '$F(N)$ of pict', nfigure=None)
                for i in range(maxcycle):
                    where_pict = np.where(index_picture_m[i] == 1)[0]
                    ax.plot(self.config.Lw_0 + m_ext[i][where_pict], m_f[i][where_pict], '.')
                save = self.to_save_fig + 'pict_ds_m'
                plot.fioritures(ax, fig, title='picture dans m', label=None, grid=None, save=save)

        if self.saving_step:
            Cell(self.tosave_segmforce, maxcycle, c_f, 'cell')
            Cell(self.tosave_segmtime, maxcycle, c_t, 'cell')
            Cell(self.tosave_segmext, maxcycle, c_ext, 'cell')

            np.save(self.tosave_indice_lw1, indice_lw1)
            np.save(self.tosave_Lw_f_ref, Lw_f_ref)
            np.save(self.tosave_f_ref, f_ref)

            Cell(self.tosave_mtt_f, maxcycle, mtt_f, 'cell')
            Cell(self.tosave_mtt_t, maxcycle, mtt_t, 'cell')
            Cell(self.tosave_mtt_trenorm, maxcycle, mtt_trenorm, 'cell')
            Cell(self.tosave_mtt_ext, maxcycle, mtt_ext, 'cell')
            np.save(self.tosave_mtt_size, indice_lw1)

            Cell(self.tosave_m1_f, maxcycle, m1_f, 'cell')
            Cell(self.tosave_m1_t, maxcycle, m1_t, 'cell')
            Cell(self.tosave_m1_trenorm, maxcycle, m1_trenorm, 'cell')
            Cell(self.tosave_m1_ext, maxcycle, m1_ext, 'cell')
            np.save(self.tosave_m1_size, m_size)

            Cell(self.tosave_m_f, maxcycle, m_f, 'cell')
            Cell(self.tosave_m_t, maxcycle, m_t, 'cell')
            Cell(self.tosave_m_trenorm, maxcycle, m_trenorm, 'cell')
            Cell(self.tosave_m_ext, maxcycle, m_ext, 'cell')
            np.save(self.tosave_m_size, m_size)

            Cell(self.tosave_index_picture_m, maxcycle, index_picture_m, 'cell')
            Cell(self.tosave_number_picture_m, maxcycle, number_picture_m, 'cell')
            np.save(self.tosave_nb_index_picture_m, nb_index_picture_m)

        if self.config.img:
            where0 = np.where(index_picture_m[0] == 1)
            whereend = np.where(index_picture_m[-1] == 1)
            print(number_picture_m[0][where0][0], number_picture_m[-1][whereend][-1])

        return indice_lw1, Lw_f_ref, f_ref, m_f, m_trenorm, m_ext, m_size, index_picture_m, number_picture_m


# ---------------------------------------------------------------------------------------------------------------------#
class Flu():
    ''' extrait fluctuation autour de la moyenne  '''

    # ---------------------------------------------------------#
    def __init__(self, config, remote, display_figure=True, saving_step=True):
        ''' nb_processes : nombre de coeurs à utiilisé pour le mutiprocessing
            display_figure : affiche les figures pour vérifier pas de pb dans l'analyse
            saving step : pêrmet de sauver les étapes dans cas de analyse signel set, si multi set alors ne sauve pas'''

        ## Config
        self.config = config
        self.remote = remote

        self.sursample = config.sursample
        self.nb_processes = config.nb_process
        self.display_figure = display_figure
        self.saving_step = saving_step

        self.to_save_flu = self.config.global_path_save + 'flu/'
        self.to_save_fig = self.config.global_path_save + 'figure_flu/'

        ## config
        if not self.config.mix_set:
            self.maxcycle = self.config.maxcycle
            self.nbcycle = self.config.nbcycle
            self.mincycle = self.config.mincycle
        else:
            self.maxcycle = np.sum(self.config.maxcycle)
            self.nbcycle = np.sum(self.config.nbcycle)
            self.mincycle = self.config.mincycle[0]
            self.sub_cycles = self.config.sub_cycles

        ## obj créés
        if self.saving_step:
            self.tosave_Lw_0_flu = self.to_save_flu + 'Lw_0_flu'
            self.tosave_beta = self.to_save_flu + 'beta'

            self.tosave_flu = self.to_save_flu + 'flu'
            self.tosave_ext_flu = self.to_save_flu + 'ext_flu'
            self.tosave_t_flu = self.to_save_flu + 't_flu'
            self.tosave_flu_size = self.to_save_flu + 'size_flu'

            self.tosave_index_picture_flu = self.to_save_flu + 'index_picture_flu'
            self.tosave_number_picture_flu = self.to_save_flu + 'number_picture_flu'

            self.tosave_nb_index_picture_flu = self.to_save_flu + 'nb_index_picture_flu'

    # ------------------------------------------
    def model(self, b, x):

        return b[0] + b[1] * x

    # ------------------------------------------
    def residus(self, b, x, y, n):

        if np.isnan(x).any():
            print('x problem', n)
        if np.isnan(y).any():
            print('y=f(i) problem', n)
        z = self.model(b, x)
        if np.isnan(z).any():
            print('model problem', n)

            where = np.where(np.isnan(z))

        if np.isnan(y - self.model(b, x)).any():
            print('res residus problem', n)

        return y - self.model(b, x)

    # ------------------------------------------
    def fit(self, n, beta0, x, y):

        resultat = opt.least_squares(self.residus, beta0, args=(x[n], y[n], n))
        beta = resultat.x
        np.save(self.tosave_beta + '_%d' % (n + 1), beta)

    # ------------------------------------------
    def main_run(self, plot, m_f, m_ext, m_t, index_picture_m, number_picture_m, fit_done=False):
        ''' étape 1 : initalisation à zero de f,t,ext
        étape 2 : fit avec model trouvé par algo JP
        étape 3 : extraire fluctuations
        fit_done : si False => trouves en multi process les paramètrre de fit par regression non linéaire et les save ;
        si fit_done = True => récupère les paramètres sauvés '''

        lw1, nb_point = self.config.config_flu()

        ## initialisation
        f = [0 for i in range(self.maxcycle)]
        ext = [0 for i in range(self.maxcycle)]
        t = [0 for i in range(self.maxcycle)]
        Lw_0_flu = np.zeros(self.maxcycle)
        if self.config.img:
            index_picture = [0 for i in range(self.maxcycle)]
            number_picture = [0 for i in range(self.maxcycle)]

        for i in range(self.maxcycle):
            f[i] = m_f[i] - m_f[i][0]
            ext[i] = m_ext[i] - np.min(m_ext[i])
            t[i] = m_t[i] - m_t[i][0]

            j = np.where(ext[i] >= lw1)[0][0]
            f[i] = f[i][j::]
            ext[i] = ext[i][j::] - ext[i][j]
            t[i] = t[i][j::] - t[i][j]
            Lw_0_flu[i] = self.config.Lw_0 + m_ext[i][j]
            if self.config.img:
                index_picture[i] = index_picture_m[i][j::]
                number_picture[i] = number_picture_m[i][j::]

        if self.display_figure:
            fig, ax = plot.belleFigure('$L_{w} (mm)$', '$F(N)$', nfigure=None)
            for i in range(self.maxcycle):
                ax.plot(Lw_0_flu[i] + ext[i], f[i])
            save = self.to_save_fig + 'F_vs_Lw'
            plot.fioritures(ax, fig, title='F vs Lw', label=None, grid=None, save=save)

            plot.plot_x_y_multiliste(self.maxcycle, t, f, 't (s)', 'F(N)', 't', 'F',
                                          save=self.to_save_fig,
                                          title='f vs t')

            if self.config.img:
                fig, ax = plot.belleFigure('$L_{w} (mm)$ of pict', '$F(N)$ of pict', nfigure=None)
                for i in range(self.maxcycle):
                    where_pict = np.where(index_picture[i] == 1)[0]
                    ax.plot(self.config.Lw_0 + ext[i][where_pict], f[i][where_pict], '.')
                save = self.to_save_fig + 'pict_ds_m'
                plot.fioritures(ax, fig, title='picture dans m', label=None, grid=None, save=save)

        ## gestion sursample
        if self.sursample:
            ## resolve resolution instron

            sort_f = [0 for i in range(self.maxcycle)]
            sort_ext = [0 for i in range(self.maxcycle)]
            f_size = [0 for i in range(self.maxcycle)]

            for i in range(self.maxcycle):
                sorted_index = np.argsort(ext[i])
                sort_f[i] = f[i][sorted_index]
                sort_ext[i] = ext[i][sorted_index]
                f_size[i] = sort_f[i].size

            f = [0 for i in range(self.maxcycle)]
            ext = [0 for i in range(self.maxcycle)]

            for i in range(self.maxcycle):
                newf = np.zeros(f_size[i] - (2 * nb_point))
                newext = np.zeros(f_size[i] - (2 * nb_point))

                for j in range(f_size[i] - (2 * nb_point)):
                    newf[j] = np.mean(sort_f[i][j:j + 2 * nb_point + 1])
                    newext[j] = np.mean(sort_ext[i][j:j + 2 * nb_point + 1])
                    if np.size(sort_f[i][j:j + 2 * nb_point + 1]) == 0:
                        print('f problem en', j)
                    if np.size(sort_ext[i][j:j + 2 * nb_point + 1]) == 0:
                        print('ext problem en', j)
                f[i] = newf
                ext[i] = newext - newext[0]
                t[i] = t[i][nb_point:-nb_point] - t[i][0]
                Lw_0_flu[i] = Lw_0_flu[i] + newext[0]

        ## fit
        X_nlin = [0 for i in range(self.maxcycle)]

        for i in range(self.maxcycle):
            X_nlin[i] = ext[i] / ext[i][-1]

        beta0 = np.array([1, 1])

        if not fit_done:
            if __name__ == "Datas.classScalar":
                with Pool(processes=self.nb_processes) as pool:
                    freeze_support()
                    pool.map(partial(self.fit, beta0=beta0, x=X_nlin, y=f), range(self.maxcycle))
            print('hey je suis sortie du multipross')
        else:
            print('param already found')

        recup_beta = Cell(self.tosave_beta, self.maxcycle)
        beta = recup_beta.reco_cell()


        if self.display_figure and not self.config.mix_set:
            for i in range(self.nbcycle):
                y = np.zeros_like(f[i])
                for j in range(f[i].size):
                    y[j] = self.model(beta[i], X_nlin[i][j])

                fig, ax = plot.belleFigure('$L_w$', '$F(N)$', nfigure=None)
                ax.plot(ext[i], f[i], 'b')
                ax.plot(ext[i], y, 'r')
                save = self.to_save_fig + 'fit_mean_F_{}'.format(i)
                plot.fioritures(ax, fig, title='moyenne de F', label=None, grid=None, save=save)

        tendance = [0 for i in range(self.maxcycle)]

        for i in range(self.maxcycle):
            x = X_nlin[i]
            tendance[i] = np.zeros_like(f[i])
            for j in range(f[i].size):
                tendance[i][j] = self.model(beta[i], x[j])

        ## fluctuations

        t_cut = [0 for i in range(self.nbcycle)]
        ext_cut = [0 for i in range(self.nbcycle)]
        f_cut = [0 for i in range(self.nbcycle)]
        tendance_cut = [0 for i in range(self.nbcycle)]
        cut_size = np.zeros(self.nbcycle, dtype=int)
        if self.config.img:
            index_picture_cut = [0 for i in range(self.nbcycle)]
            number_picture_cut = [0 for i in range(self.nbcycle)]

        coeff_a = np.zeros(self.nbcycle)
        coeff_b = np.zeros(self.nbcycle)

        if self.config.mix_set:
            l = 0

            for i in range(self.config.nb_set):

                for k in range(self.config.nbcycle[i]):
                    where_save = k + l
                    where_take = k + l + self.config.mincycle[i] - 1 + i

                    t_cut[where_save] = t[where_take]
                    ext_cut[where_save] = ext[where_take]
                    f_cut[where_save] = f[where_take]
                    tendance_cut[where_save] = tendance[where_take]
                    cut_size[where_save] = ext_cut[where_save].size
                    Lw_0_flu[where_save] = Lw_0_flu[where_take]
                    if self.config.img:
                        index_picture_cut[where_save] = index_picture[where_take]
                        number_picture_cut[where_save] = number_picture[where_take]

                    coeff_a[where_save] = beta[where_take][1]
                    coeff_b[where_save] = beta[where_take][0]

                l = l + self.config.nbcycle[i]

            grid = plot.make_grid(self.config)

            if self.display_figure:
                plot.plot_y(coeff_a, 'cycle', 'a', 'cycle', 'coeff_a',
                                 save=self.to_save_fig,
                                 title='coeff du fit : pente',
                                 grid=grid)

                plot.plot_y(coeff_b, 'cycle', 'b', 'cycle', 'coeff_b',
                                 save=self.to_save_fig,
                                 title='coeff du fit : origine',
                                 grid=grid)

        else:
            for i in range(self.nbcycle):
                t_cut[i] = t[i + self.mincycle - 1]
                ext_cut[i] = ext[i + self.mincycle - 1]
                f_cut[i] = f[i + self.mincycle - 1]
                tendance_cut[i] = tendance[i + self.mincycle - 1]
                cut_size[i] = ext_cut[i].size
                Lw_0_flu[i] = Lw_0_flu[i + self.mincycle - 1]
                if self.config.img:
                    index_picture_cut[i] = index_picture[i + self.mincycle - 1]
                    number_picture_cut[i] = number_picture[i + self.mincycle - 1]

                coeff_a[i] = beta[i + self.mincycle - 1][1]
                coeff_b[i] = beta[i + self.mincycle - 1][0]

            if self.display_figure:
                plot.plot_y(coeff_a, 'cycle', 'a', 'cycle', 'coeff_a',
                                 save=self.to_save_fig,
                                 title='coeff du fit : pente')

                plot.plot_y(coeff_b, 'cycle', 'b', 'cycle', 'coeff_b',
                                 save=self.to_save_fig,
                                 title='coeff du fit : origine')

        flu_size = np.min(cut_size)

        flu = np.zeros((self.nbcycle, flu_size))
        ext_flu = np.zeros((self.nbcycle, flu_size))
        t_flu = np.zeros((self.nbcycle, flu_size))
        index_picture_flu = np.zeros((self.nbcycle, flu_size))
        number_picture_flu = np.zeros((self.nbcycle, flu_size))

        for i in range(self.nbcycle):
            flu[i, :] = f_cut[i][0:flu_size] - tendance_cut[i][0:flu_size]

            ext_flu[i, :] = ext_cut[i][0:flu_size]
            t_flu[i, :] = t_cut[i][0:flu_size]

            if self.config.img:
                index_picture_flu[i, :] = index_picture_cut[i][0:flu_size]
                number_picture_flu[i, :] = number_picture_cut[i][0:flu_size]

        if self.config.img:
            if not self.config.mix_set:
                nb_index_picture_flu = 0
                for i in range(self.nbcycle):
                    nb_index_picture_flu = nb_index_picture_flu + np.sum(index_picture_flu[i, :])
            else:
                nb_index_picture_flu = np.zeros(self.config.nb_set)
                for j in range(self.config.nb_set):
                    for i in self.sub_cycles[j]:
                        nb_index_picture_flu[j] = nb_index_picture_flu[j] + np.sum(index_picture_flu[i, :])
        else:
            nb_index_picture_flu = None

        if self.display_figure:

            fig, ax = plot.belleFigure('$L_{w} (mm)$', '$\delta f (N)$', nfigure=None)
            for i in range(self.nbcycle):
                ax.plot(Lw_0_flu[i] + ext_flu[i], flu[i])
            plot.plt.title('fluctuations vs Lw')
            plot.plt.savefig(self.to_save_fig + 'flu_vs_Lw' + '.png')
            save = self.to_save_fig + 'flu_vs_Lw'
            plot.fioritures(ax, fig, title='fluctuations vs Lw', label=None, grid=None, save=save)

            plot.plot_x_y_multiarray(self.nbcycle, t_flu, flu, 't(s)', '\delta f (N)', 't', 'flu',
                                          save=None, title='fluctuations vs t', label=None,
                                          colors=None, grid=None)

            if self.config.img:
                fig, ax = plot.belleFigure('$L_{w} (mm)$ of pict', '$\delta f (N)$ of pict', nfigure=None)
                for i in range(self.nbcycle):
                    where_pict = np.where(index_picture_flu[i] == 1)[0]
                    ax.plot(Lw_0_flu[i] + ext_flu[i, where_pict], flu[i, where_pict], '.')
                save = self.to_save_fig + 'pict_ds_flu'
                plot.fioritures(ax, fig, title='picture dans flu', label=None, grid=None, save=save)

        if self.saving_step:
            np.save(self.tosave_Lw_0_flu, Lw_0_flu)

            np.save(self.tosave_flu, flu)
            np.save(self.tosave_ext_flu, ext_flu)
            np.save(self.tosave_t_flu, t_flu)
            np.save(self.tosave_flu_size, flu_size)

            np.save(self.tosave_index_picture_flu, index_picture_flu)
            np.save(self.tosave_number_picture_flu, number_picture_flu)

            np.save(self.tosave_nb_index_picture_flu, nb_index_picture_flu)

        return beta, flu, t_flu, ext_flu, flu_size, Lw_0_flu, index_picture_flu, number_picture_flu