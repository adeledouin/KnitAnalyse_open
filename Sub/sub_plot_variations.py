import numpy as np

# ------------------------------------------
def plot_variations_flu(plot, signal_flu, variations_flu):

    if signal_flu.config.mix_set:
        grid = plot.make_grid(signal_flu.config)
    else:
        grid = None

    colors = plot.make_colors(signal_flu.nbcycle)

# %% ##### Pdf sur flu #####

    plot.Pdf_linlin(variations_flu.ndim_to_1dim, variations_flu.stats_f.min, variations_flu.stats_f.max,
                         'F', '{}'.format(signal_flu.fname), signal_flu.to_save_fig,
                    label='de moyenne {} N et de variance {} N'.format(np.round(variations_flu.stats_f.mean, 4),
                                                               np.round(variations_flu.stats_f.var, 4)))

    plot.plot_pdf_linlin_multicycles(signal_flu.nbcycle,
                                     signal_flu.f, variations_flu.stats_f.min, variations_flu.stats_f.max,
                         'F', 'c_{}'.format(signal_flu.fname), signal_flu.to_save_fig,
                                     colors=colors)

# %% ##### variations sur tps tt par cycle #####

    plot.plot_variation_cycle(variations_flu, signal_flu.fname, save=signal_flu.to_save_fig,
                              label=signal_flu.fname, grid=grid)

# %% ##### variations sur fenetre tps par cycle#####

    plot.plot_variation_tps_cycle(variations_flu, signal_flu.fname, save=signal_flu.to_save_fig,
                                     colors=colors)

# %% ### coefficients flu

    plot.plot_coefficients(variations_flu, signal_flu.fname, save=signal_flu.to_save_fig,
                      label=signal_flu.fname, grid=grid)


# %% ##### variations sur cycle et fenetre temporelle par tps #####

    plot.plot_variation_tps(variations_flu, signal_flu.fname, save=signal_flu.to_save_fig,
                              label=signal_flu.fname, grid=grid)


# ------------------------------------------
def plot_variations_img(config, plot, signal_img, variations_sum_vit, variations_sum_slip, variations_sum_dev,
                        variations_sum_vort, variations_sum_abs_vort, variations_vit, variations_slip,
                        variations_dev, variations_vort, variations_abs_vort):

    if config.mix_set:
        grid = plot.make_grid(signal_img.config)
    else:
        grid = None

    colors = plot.make_colors(signal_img.nbcycle)

# %% ##### Pdf sur vit #####
    seuil = 1e-7
    plot.Pdf_loglin(variations_vit.ndim_to_1dim, seuil, variations_vit.stats_f.max,
                         'vit', '{}'.format('vit'), signal_img.to_save_fig,
                    label='de moyenne {} N et de variance {} N'.format(np.round(variations_vit.stats_f.mean, 6),
                                                                            np.round(variations_vit.stats_f.var, 6)))

    plot.plot_pdf_loglin_multicycles(signal_img.nbcycle,
                                     variations_vit.f_c, variations_vit.stats_f.min, variations_vit.stats_f.max,
                               'vit', 'c_{}'.format('vit'), signal_img.to_save_fig,
                                     colors=colors)


# %% ##### Pdf sur slip #####
    seuil = 1e-5
    plot.Pdf_loglin(variations_slip.ndim_to_1dim, seuil, variations_slip.stats_f.max,
                         'slip', '{}'.format('slip'), signal_img.to_save_fig,
                    label='de moyenne {} N et de variance {} N'.format(np.round(variations_slip.stats_f.mean, 4),
                                                                            np.round(variations_slip.stats_f.var, 4)))

    plot.plot_pdf_loglin_multicycles(signal_img.nbcycle,
                                     variations_slip.f_c, seuil, variations_slip.stats_f.max,
                               'slip', 'c_{}'.format('slip'), signal_img.to_save_fig,
                                     colors=colors)

# %% ##### Pdf sur vort, abs vort et dev #####
    seuil = 1e-4
    seuil_ext = 0.1
    y_Pdf_vort, x_Pdf_vort = plot.histo.my_histo(variations_vort.ndim_to_1dim,
                                            - seuil_ext, seuil_ext,
                                            'lin', 'log', density=2, binwidth=seuil, nbbin=None)
    y_Pdf_abs_vort, x_Pdf_abs_vort = plot.histo.my_histo(variations_abs_vort.ndim_to_1dim,
                                                    - seuil_ext,
                                                    seuil_ext,
                                                    'lin', 'log', density=2, binwidth=seuil, nbbin=None)
    y_Pdf_dev, x_Pdf_dev = plot.histo.my_histo(variations_dev.ndim_to_1dim,
                                          - seuil_ext, seuil_ext,
                                          'lin', 'log', density=2, binwidth=seuil, nbbin=None)

    fig, ax = plot.belleFigure('$def$', '$Pdf(def)$', figsize=(12, 8))
    ax.plot(x_Pdf_vort, y_Pdf_vort, '.',
            label='<vort> = {} N, Var(vort) = {} N'.format(np.round(variations_vort.stats_f.mean, 6),
                                                                     np.round(variations_vort.stats_f.var, 6)))
    ax.plot(x_Pdf_abs_vort, y_Pdf_abs_vort, '.',
            label='<|vort|> = {} N, Var(|vort|) = {} N'.format(
                np.round(variations_abs_vort.stats_f.mean, 6),
                np.round(variations_abs_vort.stats_f.var, 6)))
    ax.plot(x_Pdf_dev, y_Pdf_dev, '.',
            label='<dev> = {} N, Var(dev) = {} N'.format(np.round(variations_dev.stats_f.mean, 6),
                                                                    np.round(variations_dev.stats_f.var, 6)))
    plot.plt.yscale('log')
    save = signal_img.to_save_fig + 'pdf_' + 'def'
    plot.fioritures(ax, fig, title='Pdf sur def', label=True, grid=None, save=save)

    plot.plot_pdf_linlog_multicycles(signal_img.nbcycle,
                                     variations_vort.f_c, - seuil_ext, seuil_ext,
                               'vort', 'c_{}'.format('vort'), signal_img.to_save_fig,
                                     colors=colors)

    plot.plot_pdf_linlog_multicycles(signal_img.nbcycle,
                                     variations_abs_vort.f_c, variations_abs_vort.stats_f.min, seuil_ext,
                               'abs(vort)', 'c_{}'.format('abs(vort)'), signal_img.to_save_fig,
                                     colors=colors)

    plot.plot_pdf_linlog_multicycles(signal_img.nbcycle,
                                     variations_dev.f_c, variations_dev.stats_f.min, seuil_ext,
                               'dev', 'c_{}'.format('dev'), signal_img.to_save_fig,
                                     colors=colors)


# %% ##### Pdf sur sum vit #####
    plot.Pdf_linlin(variations_sum_vit.ndim_to_1dim, variations_sum_vit.stats_f.min, variations_sum_vit.stats_f.max,
                               '\Sigma_{img}vit', '{}'.format('sum_vit'), signal_img.to_save_fig,
                    label='de moyenne {} N et de variance {} N'.format(np.round(variations_sum_vit.stats_f.mean, 4),
                                                                            np.round(variations_sum_vit.stats_f.var, 4)))

    plot.plot_pdf_linlin_multicycles(signal_img.nbcycle,
                                     variations_sum_vit.f_c, variations_sum_vit.stats_f.min, variations_sum_vit.stats_f.max,
                               '\Sigma_{img}vit', 'c_{}'.format('sum_vit'), signal_img.to_save_fig,
                                     colors=colors, nbbin=25)

# %% ##### Pdf sur sum slip #####
    plot.Pdf_linlin(variations_sum_slip.ndim_to_1dim, variations_sum_slip.stats_f.min,
                    variations_sum_slip.stats_f.max,
                         '\Sigma_{img}slip', '{}'.format('sum_slip'), signal_img.to_save_fig,
                    label='de moyenne {} N et de variance {} N'.format(
                             np.round(variations_sum_slip.stats_f.mean, 4),
                             np.round(variations_sum_slip.stats_f.var, 4)))

    plot.plot_pdf_linlin_multicycles(signal_img.nbcycle,
                                     variations_sum_slip.f_c, variations_sum_slip.stats_f.min,
                                     variations_sum_slip.stats_f.max,
                               '\Sigma_{img}slip', 'c_{}'.format('sum_slip'), signal_img.to_save_fig,
                                     colors=colors, nbbin=25)

# %% ##### Pdf sur sum vort #####
    plot.Pdf_linlog(variations_sum_vort.ndim_to_1dim, variations_sum_vort.stats_f.min,
                    variations_sum_vort.stats_f.max,
                         '\Sigma_{img}vort', '{}'.format('sum_vort'), signal_img.to_save_fig,
                    label='de moyenne {} N et de variance {} N'.format(
                             np.round(variations_sum_vort.stats_f.mean, 4),
                             np.round(variations_sum_vort.stats_f.var, 4)), nbbin=150)

    plot.plot_pdf_linlog_multicycles(signal_img.nbcycle,
                                     variations_sum_vort.f_c, variations_sum_vort.stats_f.min,
                                     variations_sum_vort.stats_f.max,
                               '\Sigma_{img}vort', 'c_{}'.format('sum_vort'), signal_img.to_save_fig,
                                     colors=colors, nbbin=50)

# %% ##### Pdf sur sum abs_vort #####
    plot.Pdf_linlog(variations_sum_abs_vort.ndim_to_1dim, variations_sum_abs_vort.stats_f.min,
                    variations_sum_abs_vort.stats_f.max,
                         '\Sigma_{img}abs(vort)', '{}'.format('sum_abs_vort'), signal_img.to_save_fig,
                    label='de moyenne {} N et de variance {} N'.format(
                             np.round(variations_sum_abs_vort.stats_f.mean, 4),
                             np.round(variations_sum_abs_vort.stats_f.var, 4)), nbbin=150)

    plot.plot_pdf_linlog_multicycles(signal_img.nbcycle,
                                     variations_sum_abs_vort.f_c, variations_sum_abs_vort.stats_f.min,
                                     variations_sum_abs_vort.stats_f.max,
                               '\Sigma_{img}abs(vort)', 'c_{}'.format('sum_abs_vort'), signal_img.to_save_fig,
                                     colors=colors, nbbin=50)

# %% ##### Pdf sur sum dev #####
    plot.Pdf_linlog(variations_sum_dev.ndim_to_1dim, variations_sum_dev.stats_f.min,
                    variations_sum_dev.stats_f.max,
                         '\Sigma_{img}dev', '{}'.format('sum_dev'), signal_img.to_save_fig,
                    label='de moyenne {} N et de variance {} N'.format(
                             np.round(variations_sum_dev.stats_f.mean, 4),
                             np.round(variations_sum_dev.stats_f.var, 4)), nbbin=150)

    plot.plot_pdf_linlog_multicycles(signal_img.nbcycle,
                                     variations_sum_dev.f_c, variations_sum_dev.stats_f.min,
                                     variations_sum_dev.stats_f.max,
                               '\Sigma_{img}dev', 'c_{}'.format('sum_dev'), signal_img.to_save_fig,
                                     colors=colors, nbbin=50)

# %% ##### variations sur tps tt par cycle #####

    plot.plot_variation_cycle(variations_sum_vit, 'sum_vit', save=signal_img.to_save_fig,
                              label='sum_vit', grid=grid)

    plot.plot_variation_cycle(variations_sum_slip, 'sum_slip', save=signal_img.to_save_fig,
                              label='sum_slip', grid=grid)

    plot.plot_variation_cycle(variations_sum_vort, 'sum_vort', save=signal_img.to_save_fig,
                              label='sum_vort', grid=grid)

    plot.plot_variation_cycle(variations_sum_abs_vort, 'sum_abs_vort', save=signal_img.to_save_fig,
                              label='sum_abs_vort', grid=grid)

    plot.plot_variation_cycle(variations_sum_dev, 'sum_dev', save=signal_img.to_save_fig,
                              label='sum_dev', grid=grid)

    plot.plot_variation_cycle(variations_vit, 'vit', save=signal_img.to_save_fig,
                              label='vit', grid=grid)

    plot.plot_variation_cycle(variations_slip, 'slip', save=signal_img.to_save_fig,
                              label='slip', grid=grid)

    plot.plot_variation_cycle(variations_vort, 'vort', save=signal_img.to_save_fig,
                              label='vort', grid=grid)

    plot.plot_variation_cycle(variations_abs_vort, 'abs_vort', save=signal_img.to_save_fig,
                              label='abs_vort', grid=grid)

    plot.plot_variation_cycle(variations_dev, 'dev', save=signal_img.to_save_fig,
                              label='dev', grid=grid)

# %% ##### variations sur fenetre tps par cycle#####

    plot.plot_variation_tps_cycle(variations_sum_vit, 'sum_vit', save=signal_img.to_save_fig,
                                     colors=colors)

    plot.plot_variation_tps_cycle(variations_sum_slip, 'sum_slip', save=signal_img.to_save_fig,
                                     colors=colors)

    plot.plot_variation_tps_cycle(variations_sum_vort, 'sum_vort', save=signal_img.to_save_fig,
                                     colors=colors)

    plot.plot_variation_tps_cycle(variations_sum_abs_vort, 'sum_abs_vort', save=signal_img.to_save_fig,
                                     colors=colors)

    plot.plot_variation_tps_cycle(variations_sum_dev, 'sum_dev', save=signal_img.to_save_fig,
                                     colors=colors)

# %% ### coefficients

    plot.plot_coefficients(variations_sum_vit, 'sum_vit', save=signal_img.to_save_fig,
                      label='sum_vit', grid=grid)

    plot.plot_coefficients(variations_sum_slip, 'sum_slip', save=signal_img.to_save_fig,
                      label='sum_slip', grid=grid)

    plot.plot_coefficients(variations_sum_vort, 'sum_vort', save=signal_img.to_save_fig,
                      label='sum_vort', grid=grid)

    plot.plot_coefficients(variations_sum_abs_vort, 'sum_abs_vort', save=signal_img.to_save_fig,
                      label='sum_abs_vort', grid=grid)

    plot.plot_coefficients(variations_sum_dev, 'sum_dev', save=signal_img.to_save_fig,
                      label='sum_dev', grid=grid)


# %% ##### variations sur cycle et fenetre temporelle par tps #####

    plot.plot_variation_tps(variations_sum_vit, 'sum_vit', save=signal_img.to_save_fig,
                            label='sum_vit', grid=grid)

    plot.plot_variation_tps(variations_sum_slip, 'sum_slip', save=signal_img.to_save_fig,
                            label='sum_slip', grid=grid)

    plot.plot_variation_tps(variations_sum_vort, 'sum_vort', save=signal_img.to_save_fig,
                            label='sum_vort', grid=grid)

    plot.plot_variation_tps(variations_sum_abs_vort, 'sum_abs_vort', save=signal_img.to_save_fig,
                            label='sum_abs_vort', grid=grid)

    plot.plot_variation_tps(variations_sum_dev, 'sum_dev', save=signal_img.to_save_fig,
                            label='sum_dev', grid=grid)


