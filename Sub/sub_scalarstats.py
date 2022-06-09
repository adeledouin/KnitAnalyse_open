# %%
import numpy as np
import timeit
import scipy.optimize
from scipy import signal

from classConfig import Config
from Datas.classSignal import SignalForce, SignalImg, VariationsScalar
from Datas.classEvent_decades import ForceEvent
from Utils.classStat import Histo, Stat
from Utils.classPlot import ClassPlot
from Sub.sub_plot_variations import plot_variations_flu
from Sub.sub_plot_variations_event import plot_pdf_event, plot_variations_df_tt, plot_pdf_event_comparaison, \
    plot_variations_df_img
import Config_exp
import Config_plot


################### Main code ##################################

def scalarstats(config, signaltype, NN_data, plot, signal_flu):

    # %% ################### Partie 1 : mean et variance for all manip ##################################

    print('signal flu: {}'.format(signal_flu.f.shape))

    if config.mix_set:
        nb_set = config.nb_set
        colors = plot.make_colors(nb_set)
        grid = plot.make_grid(config)
        if NN_data == '':
            nb_sub_cycles = np.array([signal_flu.sub_cycles[i].size for i in range(nb_set)])
            sub_cycles = signal_flu.sub_cycles
            ref_sub_cycles = signal_flu.sub_cycles
        else:
            nb_sub_cycles = np.array([signal_flu.NN_sub_cycles[i].size for i in range(nb_set)])
            sub_cycles = signal_flu.NN_sub_cycles
            ref_sub_cycles = signal_flu.sub_cycles_NN


        stat_t = np.ones((nb_set, signal_flu.f.shape[1], 3)) * np.nan
        stat_c = np.ones((nb_set, np.max(nb_sub_cycles), 3)) * np.nan
        mean_stat_t = np.zeros((nb_set, 3))
        mean_stat_c = np.zeros((nb_set, 3))

        for num_set in range(nb_set):

            sub_flu = signal_flu.f[sub_cycles[num_set], :]

            if sub_flu.size != 0:
                stat_flu_t = Stat(config, sub_flu, axis=0)
                stat_flu_c = Stat(config, sub_flu, axis=1)

                stat_t[num_set, :, 0] = stat_flu_t.mean
                stat_t[num_set, :, 1] = stat_flu_t.var
                stat_t[num_set, :, 2] = stat_flu_t.sigma
                stat_c[num_set, 0:sub_cycles[num_set].size, 0] = stat_flu_c.mean
                stat_c[num_set, 0:sub_cycles[num_set].size, 1] = stat_flu_c.var
                stat_c[num_set, 0:sub_cycles[num_set].size, 2] = stat_flu_c.sigma

                mean_stat_t[num_set, 0] = np.nanmean(stat_flu_t.mean)
                mean_stat_c[num_set, 0] = np.nanmean(stat_flu_c.mean)
                mean_stat_t[num_set, 1] = np.nanmean(stat_flu_t.var)
                mean_stat_c[num_set, 1] = np.nanmean(stat_flu_c.var)
                mean_stat_t[num_set, 2] = np.nanmean(stat_flu_t.sigma)
                mean_stat_c[num_set, 2] = np.nanmean(stat_flu_c.sigma)

        color = ['C0', 'C1', 'C2']
        fig, ax = plot.belleFigure('{}'.format('set'), '{}'.format('stats'), nfigure=None)
        ax.plot(np.arange(nb_set), mean_stat_t[:, 0], '*', color=color[0], label='mean on c')
        ax.plot(np.arange(nb_set), mean_stat_t[:, 1], '*', color=color[1], label='var on c')
        # ax.plot(np.arange(nb_set), mean_stat_t[:, 2], '*', color=color[2], label='sigma on c')
        ax.plot(np.arange(nb_set), mean_stat_c[:, 0], '.', color=color[0], label='mean on t')
        ax.plot(np.arange(nb_set), mean_stat_c[:, 1], '.', color=color[1], label='var on t')
        # ax.plot(np.arange(nb_set), mean_stat_c[:, 2], '.', color=color[2], label='sigma on t')
        save = signal_flu.to_save_fig + 'mean_stats'
        plot.fioritures(ax, fig, title=None, label=True, grid=None,
                        save=save)

        yname = 'f'
        ysave = signal_flu.fname
        fig, ax = plot.belleFigure('c', '${}(c) = <{}(c,t)>_t$'.format(yname, yname), nfigure=None)
        for i in range(nb_set):
            ax.plot(np.arange(np.max(nb_sub_cycles)), stat_c[i, :, 0], '.', color=colors[i])
        save = signal_flu.to_save_fig + 'mean_c_set_' + signal_flu.fname
        plot.fioritures(ax, fig, title='moyenne sur tps par c par set', label=True, grid=None,
                        save=save)

        fig, ax = plot.belleFigure('c', '${}(c) = <{}(c,t)>_t$'.format(yname, yname), nfigure=None)
        for i in range(nb_set):
            ax.plot(ref_sub_cycles[i], stat_c[i, 0:ref_sub_cycles[i].size, 0], 'C0.')
        save = signal_flu.to_save_fig + 'mean_c_' + signal_flu.fname
        plot.fioritures(ax, fig, title='moyenne sur tps par c', label=True, grid=grid, save=save)

        fig, ax = plot.belleFigure('c', '${}(c) = Var({}(c,t))_t$'.format(yname, yname), nfigure=None)
        for i in range(nb_set):
            ax.plot(np.arange(np.max(nb_sub_cycles)), stat_c[i, :, 1], '.', color=colors[i])
        save = signal_flu.to_save_fig + 'var_c_set_' + signal_flu.fname
        plot.fioritures(ax, fig, title='var sur tps par c par set', label=True, grid=None,
                        save=save)

        fig, ax = plot.belleFigure('c', '${}(c) = Var({}(c,t))_t$'.format(yname, yname), nfigure=None)
        for i in range(nb_set):
            ax.plot(ref_sub_cycles[i], stat_c[i, 0:ref_sub_cycles[i].size, 1], 'C0.')
        save = signal_flu.to_save_fig + 'var_c_' + signal_flu.fname
        plot.fioritures(ax, fig, title='var sur tps par c', label=True, grid=grid, save=save)

        fig, ax = plot.belleFigure('t', '${}(t) = <{}(c,t)>_c$'.format(yname, yname), nfigure=None)
        for i in range(nb_set):
            ax.plot(np.arange(stat_t[i, :, 0].size), stat_t[i, :, 0], '.', color=colors[i])
        save = signal_flu.to_save_fig + 'mean_t_set_' + signal_flu.fname
        plot.fioritures(ax, fig, title='mean sur c par tps par set', label=True, grid=None,
                        save=save)

        fig, ax = plot.belleFigure('t', '${}(t) = Var({}(c,t))_c$'.format(yname, yname), nfigure=None)
        for i in range(nb_set):
            ax.plot(np.arange(stat_t[i, :, 1].size), stat_t[i, :, 1], '.', color=colors[i])
        save = signal_flu.to_save_fig + 'var_t_set_' + signal_flu.fname
        plot.fioritures(ax, fig, title='var sur c par tps par set', label=True, grid=None,
                        save=save)

        f, Pxx_spec = signal.welch(signal_flu.f, 25, window='hanning',  # apply a Hanning window before taking the DFT
                                   nperseg=256,  # compute periodograms of 256-long segments of x
                                   detrend=False,
                                   scaling='density')
        fig, ax = plot.belleFigure('{}'.format('frequency [Hz]'), '{}'.format('power spectral density'), nfigure=None)
        for i in range(signal_flu.f.shape[0]):
            plot.plt.semilogy(f, np.sqrt(Pxx_spec[i, :]), '.')
        plot.plt.xscale('log')
        plot.fioritures(ax, fig, title=None, label=None, grid=None, save=signal_flu.to_save_fig, major=None)

        # %% ################### Partie 2 : correlation fred for all manip ##################################

        Cfm = np.zeros((nb_set, int(signal_flu.f.shape[1] / 2)))
        inv_Cfm = np.zeros((nb_set, int(signal_flu.f.shape[1] / 2)))
        R_0_m = np.zeros(nb_set)

        for num_set in range(nb_set):
            sub_flu = signal_flu.f[sub_cycles[num_set], :]

            if sub_flu.size != 0:
                Cf = np.zeros((sub_flu.shape[0], int(sub_flu.shape[1] / 2)))
                plateaux = np.zeros(sub_flu.shape[0])
                R_0 = np.zeros(sub_flu.shape[0])
                K_0 = np.zeros(sub_flu.shape[0])
                for i in range(sub_flu.shape[0]):
                    fluctuations = sub_flu[i, :]
                    c = np.zeros(int(sub_flu.shape[1] / 2))
                    c[0] = 0
                    for j in range(1, int(sub_flu.shape[1] / 2)):
                        c[j] = 1 / 2 * np.mean((fluctuations[j::] - fluctuations[0:-j]) ** 2)

                    R_0[i] = np.mean(fluctuations ** 2)
                    plateaux[i] = np.mean(c[int(np.round(int(sub_flu.shape[1] / 2) / 3))::])
                    Cf[i, :] = c

                for i in range(int(sub_flu.shape[1] / 2)):
                    Cfm[num_set, i] = np.mean(Cf[:, i])

                R_0_m[num_set] = np.mean(R_0)
                inv_Cfm[num_set, :] = -Cfm[num_set, :] + R_0_m[num_set]

        fig, ax = plot.belleFigure('$tps (s)$', '$C(tps)$', nfigure=None)
        for num_set in range(nb_set):
            ax.plot(np.arange(int(signal_flu.f.shape[1] / 2)) * 0.04, inv_Cfm[num_set, :], '.', color=colors[num_set])
        ax.plot(np.arange(int(signal_flu.f.shape[1] / 2)) * 0.04, np.zeros(int(signal_flu.f.shape[1] / 2)), 'r')
        save = signal_flu.to_save_fig + 'correlations_' + signal_flu.fname
        plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save)

        # %% ################### Partie 3 : Variation flu ##################################

        yname = 'f(c, t)'
        ysave = 'f'
        title = True
        label = True
        fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
        y_Pdf, x_Pdf = plot.histo.my_histo(signal_flu.f.reshape(signal_flu.f.size),
                                           np.min(signal_flu.f),
                                           np.max(signal_flu.f),
                                           'lin', 'lin', density=1, binwidth=None, nbbin=100)

        ax.plot(x_Pdf, y_Pdf, '.')
        save = signal_flu.to_save_fig + 'Pdf_' + signal_flu.fname
        plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

        yname = 'f(c, t)'
        ysave = 'f'
        title = True
        label = True
        fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
        for num_set in range(nb_set):
            sub_flu = signal_flu.f[sub_cycles[num_set], :]

            if sub_flu.size != 0:
                y_Pdf, x_Pdf = plot.histo.my_histo(sub_flu.reshape(sub_flu.size),
                                                   np.min(sub_flu),
                                                   np.max(sub_flu),
                                                   'lin', 'lin', density=2, binwidth=None, nbbin=100)

                ax.plot(x_Pdf, y_Pdf, '.', color=colors[num_set])
        save = signal_flu.to_save_fig + 'Pdf_set_' + signal_flu.fname
        plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

    else:
        stat_t = np.ones((signal_flu.f.shape[1], 3)) * np.nan
        stat_c = np.ones((signal_flu.f.shape[0], 3)) * np.nan
        mean_stat_t = np.zeros(3)
        mean_stat_c = np.zeros(3)

        stat_flu_t = Stat(config, signal_flu.f, axis=0)
        stat_flu_c = Stat(config, signal_flu.f, axis=1)

        stat_t[:, 0] = stat_flu_t.mean
        stat_t[:, 1] = stat_flu_t.var
        stat_t[:, 2] = stat_flu_t.sigma
        stat_c[:, 0] = stat_flu_c.mean
        stat_c[:, 1] = stat_flu_c.var
        stat_c[:, 2] = stat_flu_c.sigma

        color = ['C0', 'C1', 'C2']
        fig, ax = plot.belleFigure('{}'.format('cycle'), '{}'.format('stats c'), nfigure=None)
        ax.plot(np.arange(config.nbcycle), stat_c[:, 0], '.', label='mean on c')
        ax.plot(np.arange(config.nbcycle), stat_c[:, 1], '.', label='var on c')
        save = None
        plot.fioritures(ax, fig, title=None, label=True, grid=None,
                        save=save)

        f, Pxx_spec = signal.welch(signal_flu.f, 25, window='hanning',  # apply a Hanning window before taking the DFT
                                   nperseg=256,  # compute periodograms of 256-long segments of x
                                   detrend=False,
                                   scaling='density')
        fig, ax = plot.belleFigure('{}'.format('frequency [Hz]'), '{}'.format('power spectral density'), nfigure=None)
        for i in range(signal_flu.f.shape[0]):
            plot.plt.semilogy(f, np.sqrt(Pxx_spec[i, :]), '.')
        plot.plt.xscale('log')
        plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

        # %% ################### Partie 2 : correlation fred for all manip ##################################

        Cfm = np.zeros((int(signal_flu.f.shape[1] / 2)))
        inv_Cfm = np.zeros((int(signal_flu.f.shape[1] / 2)))

        Cf = np.zeros((signal_flu.f.shape[0], int(signal_flu.f.shape[1] / 2)))
        plateaux = np.zeros(signal_flu.f.shape[0])
        R_0 = np.zeros(signal_flu.f.shape[0])
        K_0 = np.zeros(signal_flu.f.shape[0])
        for i in range(signal_flu.f.shape[0]):
            fluctuations = signal_flu.f[i, :]
            c = np.zeros(int(signal_flu.f.shape[1] / 2))
            c[0] = 0
            for j in range(1, int(signal_flu.f.shape[1] / 2)):
                c[j] = 1 / 2 * np.mean((fluctuations[j::] - fluctuations[0:-j]) ** 2)

            R_0[i] = np.mean(fluctuations ** 2)
            plateaux[i] = np.mean(c[int(np.round(int(signal_flu.f.shape[1] / 2) / 3))::])
            Cf[i, :] = c

        for i in range(int(signal_flu.f.shape[1] / 2)):
            Cfm[i] = np.mean(Cf[:, i])

        R_0_m = np.mean(R_0)
        inv_Cfm = -Cfm + R_0_m

        fig, ax = plot.belleFigure('$tps (s)$', '$C(tps)$', nfigure=None)
        ax.plot(np.arange(int(signal_flu.f.shape[1] / 2)) * 0.04, inv_Cfm, '.')
        ax.plot(np.arange(int(signal_flu.f.shape[1] / 2)) * 0.04, np.zeros(int(signal_flu.f.shape[1] / 2)), 'r')
        save = None
        plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save)

        # %% ################### Partie 3 : Variation flu ##################################

        print('------- variations de flu -------')

        variations_flu = VariationsScalar(config, pourcentage=5, f=signal_flu.f, ext=signal_flu.ext, t=signal_flu.t,
                                          index=None, number=None, directsignal=True,
                                          signaltype=signaltype, NN_data=NN_data, ftype='force', fname=signaltype,
                                          stats=True)

        plot_variations_flu(plot, signal_flu, variations_flu)