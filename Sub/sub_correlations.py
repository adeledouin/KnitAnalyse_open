# %%
import numpy as np

from Datas.classScalarImg import Correlations

# %% ################## main function ##################################

def correlations(config, plot, signal_flu, NN_data, display_figure):

    correlations_flu = Correlations(config, signal_flu.f, signal_flu.ext, signal_flu.t,
                                    signaltype=signal_flu.signaltype, NN_data=NN_data)

    if display_figure:
        if config.mix_set:
            grid = plot.make_grid(config)
        else:
            grid = None

        # Define the colors to be used using rainbow map (or any other map)
        colors = plot.make_colors(signal_flu.nbcycle)

        print('---- valeaur du plateau moyen est {} ----'.format(correlations_flu.plateau_mean_C))

        plot.plot_x_y_multiarray(signal_flu.nbcycle, correlations_flu.eps_C_c, correlations_flu.C_c,
                                 '\epsilon (mm)', 'C(\epsilon)',
                                 'eps{}'.format(correlations_flu.savename), 'C_c',
                                 save=correlations_flu.to_save_fig, title='correlation vs eps', colors=colors)

        plot.plot_x_y(correlations_flu.eps_mean_C, correlations_flu.mean_C,
                      '\epsilon (mm)', 'C(\epsilon)',
                      'eps{}'.format(correlations_flu.savename), 'mean_C',
                      save=correlations_flu.to_save_fig)

        fig, ax = plot.belleFigure('$\epsilon (mm)$', '$C(\epsilon)$', nfigure=None)
        ax.plot(correlations_flu.eps_mean_C, correlations_flu.inv_mean_C, 'b')
        ax.plot(correlations_flu.eps_mean_C, np.zeros(correlations_flu.eps_max), 'r')
        save = correlations_flu.to_save_fig + 'inv_mean_C_vs_eps' + correlations_flu.savename
        plot.fioritures(ax, fig, title='mean correlation vs eps', label=None, grid=None, save=save)

        plot.plot_y(correlations_flu.plateau_c, 'cycle', 'plateau',
                    'cycle{}'.format(correlations_flu.savename), 'plateau',
                    save=correlations_flu.to_save_fig, title='plateaux', label=None, grid=grid)

