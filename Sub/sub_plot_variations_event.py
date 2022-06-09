import numpy as np


# ------------------------------------------
def plot_pdf_event(config_plot, plot, signalevent, df, variations_df, savename, seuil, exposant, display_figure_reg):
    plot.Pdf_loglog(df, seuil, variations_df.stats_f.max,
                    '\Delta \delta f', 'df' + savename,
                    save=signalevent.to_save_fig,
                    label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(exposant) + 'N')

    plot.Pdf_loglog(df, seuil, variations_df.stats_f.max,
                    '\Delta \delta f', 'df' + savename,
                    save=signalevent.to_save_fig,
                    label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(exposant) + 'N',
                    defreg=True)

    if display_figure_reg:
        plot.plot_reg(df, seuil, variations_df.stats_f.max,
                      '\Delta \delta f', 'Pdf(\Delta \delta f)',
                      'pdf', 'df' + savename,
                      minreg=config_plot[
                          'signalevent_pdf(df' + savename + ')_minreg'],
                      maxreg=config_plot[
                          'signalevent_pdf(df' + savename + ')_maxreg'],
                      save=signalevent.to_save_fig,
                      label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(
                          exposant) + 'N',
                      powerlaw=-1.5)
        print(config_plot[
                          'signalevent_pdf(df' + savename + ')_minreg'], config_plot[
                          'signalevent_pdf(df' + savename + ')_maxreg'])

# -----------------------------------------
def plot_variations_df_tt(plot, signalevent, variations_df_tt, i, seuils, exposants, display_figure_reg):

    if signalevent.config.mix_set:
        grid = plot.make_grid(signalevent.config)
    else:
        grid = None

    colors = plot.make_colors(signalevent.nbcycle)

    plot.plot_pdf_loglog_multicycles(signalevent.nbcycle, variations_df_tt.f_c, seuils[i], variations_df_tt.stats_f.max,
                               'df_{tt}', 'c_df' + signalevent.savename_df_tt + '_goodseuil',
                                     save=signalevent.to_save_fig,
                                     colors=colors)

    # %% ##### variations sur tps tt par cycle #####

    plot.plot_variation_cycle(variations_df_tt, 'df' + signalevent.savename_df_tt, save=signalevent.to_save_fig,
                              label=signalevent.fname, grid=grid)

    # %% ##### variations sur fenetre tps par cycle#####

    plot.plot_variation_tps_cycle(variations_df_tt, 'df' + signalevent.savename_df_tt, save=signalevent.to_save_fig,
                                  colors=colors)

# ------------------------------------------
def plot_pdf_event_comparaison(plot, df1, df2, variations_df1, variations_df2, yname, ysave, seuil, exposant,
                               xaxis, yaxis, save=None):

    if xaxis == 'log':
        mindf1 = np.min(df1[df1 != 0])
        mindf2 = np.min(df2[df2 != 0])
    else :
        mindf1 = variations_df1.stats_f.min
        mindf2 = variations_df2.stats_f.min

    y_Pdf_df_tt, x_Pdf_df_tt = plot.histo.my_histo(df1, mindf1,
                                                    variations_df1.stats_f.max, xaxis, yaxis,
                                                    density=2, binwidth=None, nbbin=100)

    y_Pdf_df_seuil, x_Pdf_df_seuil = plot.histo.my_histo(df2,  mindf2,
                                                          variations_df2.stats_f.max, xaxis, yaxis,
                                                          density=2, binwidth=None, nbbin=100)

    fig, ax = plot.belleFigure('${}$'.format(yname), '$Pdf({})$'.format(yname), nfigure=None)
    ax.plot(x_Pdf_df_tt, y_Pdf_df_tt, '.')
    ax.plot(x_Pdf_df_seuil, y_Pdf_df_seuil, '.',
            label='seuil = {}e-{}'.format(seuil * 10 ** exposant, exposant))
    if xaxis == 'log':
        plot.plt.xscale('log')
    if yaxis == 'log':
        plot.plt.yscale('log')
    if save is not None:
        save = save + 'pdf_' + ysave
    plot.fioritures(ax, fig, title='Pdf sur {}'.format(ysave), label=True, grid=None, save=save)

# ------------------------------------------
def plot_variations_df_img(plot, signal_img, signalevent, variations_df1, variations_df2, yname, ysave, seuil, exposant, save=None):

    if signal_img.config.mix_set:
        grid = plot.make_grid(signalevent.config)
    else:
        grid = None
        
    colors = plot.make_colors(signalevent.nbcycle)

    # %% ##### variations sur tps tt par cycle #####

    fig, ax = plot.belleFigure('c', '${}(c) = <{}(c,t)>_t$'.format(yname, yname), nfigure=None)
    ax.plot(np.arange(signal_img.nbcycle),
            [variations_df1.stats_f_cycle[i].mean for i in range(signal_img.nbcycle)], '.', label='tt')
    ax.plot(np.arange(signal_img.nbcycle),
            [variations_df2.stats_f_cycle[i].mean for i in range(signal_img.nbcycle)], '.',
            label='seuil = {}e-{}'.format(seuil * 10 ** exposant, exposant))
    if save is not None:
        save_end = save + 'stats_cycle_mean_' + ysave
    plot.fioritures(ax, fig, title='moyenne de cycle en cycle', label=True, grid=grid, save=save_end)

    fig, ax = plot.belleFigure('c', '${}(c) = Var({}(c,t))_t$'.format(yname, yname), nfigure=None)
    ax.plot(np.arange(signal_img.nbcycle),
            [variations_df1.stats_f_cycle[i].var for i in range(signal_img.nbcycle)], '.', label='tt')
    ax.plot(np.arange(signal_img.nbcycle),
            [variations_df2.stats_f_cycle[i].var for i in range(signal_img.nbcycle)], '.',
            label='seuil = {}e-{}'.format(seuil * 10 ** exposant, exposant))
    if save is not None:
        save_end = save + 'stats_cycle_var_' + ysave
    plot.fioritures(ax, fig, title='var de cycle en cycle', label=True, grid=grid, save=save_end)

# ------------------------------------------
def img_events_seuils(plot, nb_seuils, variations, yname, ysave, signal_img, seuil, exposant, nbbin, grid):
    plot.plot_pdf_loglog_multiseuil(nb_seuils, variations,
                                     yname, ysave + '_seuils',
                                     signal_img.to_save_fig, seuil, exposant,
                                     nbbin=nbbin)

    plot.plot_variation_cycle(variations, ysave + '_seuils',
                             save=signal_img.to_save_fig,
                             grid=grid, multi_seuils=True, nb_seuils=nb_seuils, seuils=seuil, exposants=exposant)

# ------------------------------------------
def img_events(plot, variations, xname, yname, xsave, ysave, signal_img, seuil, exposant, nbbin, grid, colors,
               minreg, maxreg, display_figure_reg):
    plot.Pdf_loglog(variations.f, np.min(variations.f[variations.f != 0]), variations.stats_f.max,
                    yname, ysave,
                    signal_img.to_save_fig,
                    label='seuil = {}e-{}'.format(np.str(seuil * 10 ** exposant), np.str(exposant)),
                    nbbin=nbbin)

    plot.Pdf_loglog(variations.f, np.min(variations.f[variations.f != 0]), variations.stats_f.max,
                    yname, ysave + '_defreg',
                    signal_img.to_save_fig,
                    label='seuil = {}e-{}'.format(np.str(seuil * 10 ** exposant), np.str(exposant)),
                    nbbin=nbbin, defreg=True)

    if display_figure_reg:
        plot.plot_reg(variations.f,
                      variations.stats_f.min, variations.stats_f.max,
                      yname, xname, ysave, xsave,
                      minreg, maxreg, save=signal_img.to_save_fig,
                      label='seuil = {}e-{}'.format(np.str(seuil * 10 ** exposant), np.str(exposant)))

    plot.plot_pdf_loglog_multicycles(variations.nbcycle, variations.f_c,
                                     np.min(variations.f[variations.f != 0]), variations.stats_f.max,
                                     yname, ysave, signal_img.to_save_fig, colors=colors,
                                     nbbin=25)

    plot.plot_variation_cycle(variations, ysave,
                              save=signal_img.to_save_fig,
                              label='seuil = {}e-{}'.format(np.str(seuil * 10 ** exposant), np.str(exposant)),
                              grid=grid)


# ------------------------------------------
def pdf_df_S_f(plot, histo, ysave, df, variations_df, seuil_df, exposant_df, savename,
               variations_S_f, seuils_img, exposants_img, minreg=None, maxreg=None, powerlaw=None, save=None, display_figure_reg=False):

    Y_df, X_df = histo.my_histo(df, variations_df.stats_f.min,
                                            variations_df.stats_f.max,
                                            'log', 'log', density=2, binwidth=None, nbbin=150)

    fig, ax = plot.belleFigure('$\Delta events$', '$P(\Delta events)$', nfigure=None)
    ax.plot(np.log10(X_df), np.log10(Y_df), '.',
            label='df, seuil = {}e-{}'.format(seuil_df * 10 ** exposant_df, exposant_df))
    for i in range(np.size(savename)):
        Y_Pdf_S_f, X_Pdf_S_f = histo.my_histo(variations_S_f[i].f,
                                              np.min(variations_S_f[i].f[variations_S_f[i].f != 0]),
                                              variations_S_f[i].stats_f.max,
                                              'log', 'log',
                                              density=2, binwidth=None, nbbin=100)

        ax.plot(np.log10(X_Pdf_S_f), np.log10(Y_Pdf_S_f), '.',
                label=' {}, seuil = {}e-{}'.format(savename[i], seuils_img[i] * 10 ** exposants_img[i],
                                                   exposants_img[i]))
    save_end = save + 'pdf_' + ysave
    plot.fioritures(ax, fig, title='pdf de {}'.format(ysave), label=True, grid=None, save=save_end)

    if display_figure_reg:
        linx = X_df[Y_df != 0]
        liny = Y_df[Y_df != 0]


        coef_distri, x, y = histo.regression(linx, liny, minreg, maxreg)

        polynomial = np.poly1d(coef_distri)
        ys = polynomial(x)
        if powerlaw is not None:
            ypower = powerlaw * x + coef_distri[1]

        fig, ax = plot.belleFigure('$\Delta events$', '$P(\Delta events)$', nfigure=None)
        ax.plot(np.log10(X_df), np.log10(Y_df), '.',
                label='df, seuil = {}e-{}'.format(seuil_df * 10 ** exposant_df, exposant_df))
        for i in range(np.size(savename)):
            Y_Pdf_S_f, X_Pdf_S_f = histo.my_histo(variations_S_f[i].f,
                                                  np.min(variations_S_f[i].f[variations_S_f[i].f != 0]),
                                                  variations_S_f[i].stats_f.max,
                                                  'log', 'log',
                                                  density=2, binwidth=None, nbbin=100)

            ax.plot(np.log10(X_Pdf_S_f), np.log10(Y_Pdf_S_f), '.',
                    label=' {}, seuil = {}e-{}'.format(savename[i], seuils_img[i] * 10 ** exposants_img[i],
                                                       exposants_img[i]))
        ax.plot(x, ys, 'r-', label='coeff polifit = {}'.format(coef_distri[0]))
        if powerlaw is not None:
            ax.plot(x, ypower, 'g-', label='coeff powerlaw = {}'.format(powerlaw))
        save_end = save + 'pdf_events' + ysave + '_reg'
        plot.fioritures(ax, fig, title='pdf des events', label=True, grid=None, save=save_end)