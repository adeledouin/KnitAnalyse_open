# %%
import numpy as np
import timeit

from classConfig import Config
from Datas.classSignal import SignalForce, SignalImg, VariationsScalar
from Datas.classEvent_decades import ForceEvent
from Utils.classStat import Histo
from Utils.classPlot import ClassPlot
from Sub.sub_plot_variations_event import plot_pdf_event, plot_variations_df_tt, plot_pdf_event_comparaison, \
    plot_variations_df_img
import Config_exp
import Config_plot

################### Main code ##################################

remote = False

ref_tricot = 'knit005_'
n_exp = 'mix_'
version_work = 'v1'
path_from_root = '/path_from_root/'
NAME_EXP = ref_tricot + n_exp + version_work

print(NAME_EXP)

config = Config(path_from_root, Config_exp.exp[NAME_EXP])
config_plot = Config_plot.plot[NAME_EXP]

histo = Histo(config)
plot = ClassPlot(remote, histo)

signaltype = 'flu_rsc'
NN_data = 'train'
Sm = False
display_figure = True
display_figure_reg = False
display_figure_variations = False

# %% ################### Partie 0 : Detect events ##################################
print('#-------------detect events--------------#')

signal_flu = SignalForce(config, signaltype, NN_data)

# if config.img:
#     signal_img = SignalImg(config, signaltype, NN_data, fields=False)

signalevent = ForceEvent(config, signal_flu.f, signal_flu.ext, signal_flu.t, signaltype, NN_data, Sm=Sm, display_figure_debug=False)

if display_figure:
    i = config_plot['signalevent_cycle_plot']

    plot.plot_x_y(signalevent.ext[i, :], signalevent.f[i, :], 'L_{w} (mm)', '\delta f',
                  'Lw_c{}'.format(i), signaltype, save=signalevent.to_save_fig, title='Knit-quakes noise', pts='-')

    ini = config_plot['signalevent_training_event_i']
    # fin = config_plot['signalevent_training_event_f']
    fin = ini +256

    plot.plot_x_y(signalevent.ext[i, ini:fin], signalevent.f[i, ini:fin], 'L_w (mm)', '\delta f',
                  't_c{}_zoom'.format(i), signaltype, save=signalevent.to_save_fig, title='$\delta f$')

    plot.plot_x_y(signalevent.ext[i, :], signalevent.f[i, :], 'L_{w} (mm)', '\delta f',
                  'Lw_c{}'.format(i), signaltype, save=signalevent.to_save_fig, title='$\delta f$', pts='-')

    where_events = np.where(signalevent.index_events[i, :] == 1)[0]

    fig, ax = plot.belleFigure('$L_{w} (mm)$', '$\delta f (N)$', nfigure=None)
    ax.plot(signalevent.ext[i, :], signalevent.f[i, :], '.')
    ax.plot(signalevent.ext[i, where_events], signalevent.f[i, where_events], 'r.')
    plot.plt.xlim(config_plot['signalevent_zoom_xlim'])
    save = signalevent.to_save_fig + signaltype + '_zoom_c{}'.format(i) + '.pdf'
    plot.fioritures(ax, fig, title='$\delta f$ zoom', label=None, grid=None, save=save)

def retrieve_data_plot(cycle, t0, seq_size, futur):
    timeseq = signal_flu.f[cycle, t0:t0 + seq_size]
    futur = signal_flu.f[cycle, t0 + seq_size:t0 + seq_size + futur]

    return timeseq, futur

# %% ################### Stats on delta f ##################################
major = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
decade = np.array([0, 3e-3, 3e-2, 3e-1, 3e0, 3e1])
decade_grid = np.array([3e-3, 3e-2, 3e-1, 3e0,  3e1])
log_classes = np.array([1e-4, 3e-3, 3e-2, 3e-1, 3e0, 3e1])
class_grid = log_classes
exposants = [4, 3, 2, 1, 0]
save_seuil = ['0', '3_3', '3_2', '3_0']

seuil_labels = ['seuil =' + np.str(log_classes[i] * 10 ** exposants[i]) + 'e-' + np.str(exposants[i]) + 'N' for i in range(log_classes.size -1)]

yname = '\delta f'
ysave = 'df'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.df_tt, 1e-4,
                                   np.max(signalevent.df_tt),
                                   'log', 'log', density=2, binwidth=None, nbbin=70)

ax.plot(x_Pdf, y_Pdf, '.')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None #path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=None, grid=decade_grid, save=save, major=None)

yname = '\delta f'
ysave = 'df'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
for i in range(log_classes.size-1):
    where = np.where((signalevent.df_tt >= log_classes[i]) & (signalevent.df_tt <= log_classes[i + 1]))
    y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.df_tt[where], log_classes[i] if log_classes[i] != 0 else 1e-4,
                                       np.max(signalevent.df_tt[where]),
                                       'log', 'log', density=1, binwidth=None, nbbin=70)

    ax.plot(x_Pdf, y_Pdf, '.', label=seuil_labels[i])
plot.plt.xscale('log')
plot.plt.yscale('log')
for i in range(log_classes.size):
    plot.plt.axvline(x=class_grid[i], color='k')
save = None #path_Tex + '{}'.format('pdf_df_') + num_TeX
plot.fioritures(ax, fig, title=None, label=label, grid=grid, save=save, major=major)

yname = '\delta f'
ysave = 'df'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
for i in range(log_classes.size-1):
    where = np.where((signalevent.df_tt >= log_classes[i]) & (signalevent.df_tt <= log_classes[i + 1]))
    y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.df_tt[where], log_classes[i] if log_classes[i] != 0 else 1e-4,
                                       np.max(signalevent.df_tt[where]),
                                       'log', 'log', density=1, binwidth=None, nbbin=70)

    ax.plot(x_Pdf, y_Pdf, '.')
plot.plt.xscale('log')
plot.plt.yscale('log')
for i in range(log_classes.size):
    plot.plt.axvline(x=class_grid[i], color='k')
save = None #path_Tex + '{}'.format('pdf_df_') + num_TeX
plot.fioritures(ax, fig, title=None, label=None, grid=grid, save=save, major=major)

hist, _ = np.histogram(signalevent.df_tt, bins=decade, density=False)
print('prop_on_decade in df in all train = {}'.format(np.round(hist / signalevent.df_tt.size * 100, 1)))


yname = '\delta t'
ysave = 'dt'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.dt_tt, np.min(signalevent.dt_tt),
                                   np.max(signalevent.dt_tt),
                                   'lin', 'log', density=2, binwidth=0.04, nbbin=None)

ax.plot(x_Pdf, y_Pdf, '.')
# plot.plt.xscale('lin')
plot.plt.yscale('log')
save = None #path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

yname = '\delta t'
ysave = 'dt'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
for i in range(log_classes.size-1):
    where = np.where((signalevent.df_tt >= log_classes[i]) & (signalevent.df_tt <= log_classes[i + 1]))
    y_Pdf, x_Pdf = plot.histo.my_histo(signalevent.dt_tt[where], np.min(signalevent.dt_tt[where]),
                                       np.max(signalevent.dt_tt[where]),
                                       'lin', 'log', density=1, binwidth=0.08, nbbin=None)

    ax.plot(x_Pdf, y_Pdf, '.', label=seuil_labels[i])
# plot.plt.xscale('log')
plot.plt.yscale('log')
save = None #path_Tex + '{}'.format('pdf_df_') + num_TeX
plot.fioritures(ax, fig, title=None, label=label, grid=None, save=save, major=None)

# %% #####  t btw events
print('#-------------sub signal : df--------------#')
start_time = timeit.default_timer()

df_tab = signalevent.df_tab()

hist, _ = np.histogram(df_tab.reshape(df_tab.size), bins=decade, density=False)
print('prop_on_decade in df in all train = {}'.format(np.round(hist / df_tab.size * 100, 2)))


for i in range(decade.size-2):
    df_seuil, dt_seuil, index_df_seuil, number_df_seuil = signalevent.df_seuil(decade[i], decade[i+1], df_tab)

    dt_btw, index_dt_btw, number_dt_btw = signalevent.time_btw_df(index_df_seuil, number_df_seuil)
    nb_pts = np.round(dt_btw /0.04, 0)
    if i == 0:
        add = signalevent.f.shape[0]*signalevent.f.shape[1]-signalevent.dt_tt.size
        nb_pts = np.concatenate((nb_pts, np.ones(add)))
    yname = '\delta t_{btw}'
    ysave = 'dt_btw'
    title = True
    label = True
    grid = decade_grid
    fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
    y_Pdf, x_Pdf = plot.histo.my_histo(dt_btw, np.min(dt_btw),
                                       np.max(dt_btw),
                                       'lin', 'log', density=2, binwidth=0.4, nbbin=None)

    ax.plot(x_Pdf, y_Pdf, '.')
    # plot.plt.xscale('lin')
    plot.plt.yscale('log')
    save = None  # path_Tex + '{}'.format('pdf_df_all')
    plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

    yname = 'nb pts btw'
    ysave = 'dt_btw'
    title = True
    label = True
    grid = decade_grid
    fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
    y_Pdf, x_Pdf = plot.histo.my_histo(nb_pts, np.min(nb_pts),
                                       np.max(nb_pts),
                                       'lin', 'log', density=2, binwidth=10, nbbin=None)

    ax.plot(x_Pdf, y_Pdf, '.')
    # plot.plt.xscale('lin')
    plot.plt.yscale('log')
    save = None  # path_Tex + '{}'.format('pdf_df_all')
    plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

    print('for decade {} | min t btw = {} | max t btw = {} | <t btw> = {} | s(t btw) = {}'.format(i,
                                                                                  np.round(np.min(dt_btw), 2),
                                                                                  np.round(np.max(dt_btw), 2),
                                                                                  np.round(np.mean(dt_btw), 2),
                                                                                  np.round(np.sqrt(np.var(dt_btw))), 2))

    print('for decade {} | min nb pts btw = {} | max nb pts btw = {} | <nb pts btw> = {} | s(nb pts btw) = {}'.format(i,
                                                                                                  np.min(nb_pts),
                                                                                                  np.max(nb_pts),
                                                                                                  np.mean(nb_pts),
                                                                                                  np.sqrt(
                                                                                                      np.var(nb_pts))))


i = 4
df_seuil, dt_seuil, index_df_seuil, number_df_seuil = signalevent.df_seuil(decade[i], decade[i + 1], df_tab)

dt_btw, index_dt_btw, number_dt_btw = signalevent.time_btw_df(index_df_seuil, number_df_seuil)
nb_pts = np.round(dt_btw /0.04, 0)

yname = '\delta t_{btw}'
ysave = 'dt_btw'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(dt_btw, np.min(dt_btw),
                                   np.max(dt_btw),
                                   'lin', 'log', density=2, binwidth=10, nbbin=None)

ax.plot(x_Pdf, y_Pdf, '.')
# plot.plt.xscale('lin')
plot.plt.yscale('log')
save = None  # path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

yname = 'nb pts btw'
ysave = 'dt_btw'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(nb_pts, np.min(nb_pts),
                                   np.max(nb_pts),
                                   'lin', 'log', density=2, binwidth=250, nbbin=None)

ax.plot(x_Pdf, y_Pdf, '.')
# plot.plt.xscale('lin')
plot.plt.yscale('log')
save = None  # path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=None, grid=None, save=save, major=None)

print('for decade {} | min t btw = {} | max t btw = {} | <t btw> = {} | s(t btw) = {}'.format(i,
                                                                                              np.round(np.min(dt_btw),
                                                                                                       2),
                                                                                              np.round(np.max(dt_btw),
                                                                                                       2),
                                                                                              np.round(np.mean(dt_btw),
                                                                                                       2),
                                                                                              np.round(np.sqrt(
                                                                                                  np.var(dt_btw))), 2))

print('for decade {} | min nb pts btw = {} | max nb pts btw = {} | <nb pts btw> = {} | s(nb pts btw) = {}'.format(i,
                                                                                                                  np.min(
                                                                                                                      nb_pts),
                                                                                                                  np.max(
                                                                                                                      nb_pts),
                                                                                                                  np.mean(
                                                                                                                      nb_pts),
                                                                                                                  np.sqrt(
                                                                                                                      np.var(
                                                                                                                          nb_pts))))

# %% ##### correlations
# eps_max = int(np.round((20/100) * np.size(signalevent.t[0, :])))
#
# # ------------------------------------------
# def correlations_par_cycle(f):
#     eps = signalevent.t[:, 0:eps_max]
#     nbcycle = f.shape[0]
#     C = np.zeros((nbcycle, eps_max))
#     plateaux = np.zeros(nbcycle)
#
#     for i in range(nbcycle):
#         print(i)
#         fluctuations = f[i, :]
#         c = np.zeros(eps_max)
#         c[0] = 0
#         for j in range(1, eps_max):
#             c[j] = 1 / 2 * np.mean((fluctuations[j::] - fluctuations[0:-j]) ** 2)
#
#         plateaux[i] = np.mean(c[int(np.round(eps_max / 3))::])
#         C[i, :] = c
#
#     return C, eps, plateaux
#
# # ------------------------------------------
# def mean_correlations(f):
#     mean_C = np.zeros(eps_max)
#     eps_mean_C = np.zeros(eps_max)
#
#     C_c, eps_C_c, plateau_c = correlations_par_cycle(f)
#
#     for i in range(eps_max):
#         mean_C[i] = np.mean(C_c[:, i])
#         eps_mean_C[i] = np.mean(eps_C_c[:, i])
#
#     plateau_mean_C = np.mean(mean_C[int(np.round(eps_max / 3))::])
#     inv_mean_C = -mean_C + plateau_mean_C
#
#     return mean_C, eps_mean_C, plateau_mean_C, inv_mean_C, C_c, eps_C_c, plateau_c
#
# # ------------------------------------------
# def temporale_auto_correlation_par_cycle(f):
#     eps = signalevent.t[:, 0:eps_max]
#     nbcycle = f.shape[0]
#     C = np.zeros((nbcycle, eps_max))
#     plateaux = np.zeros(nbcycle)
#
#     for i in range(nbcycle):
#         print(i)
#         fluctuations = f[i, :]
#         c = np.zeros(eps_max)
#         c[0] = 0
#         for j in range(1, eps_max):
#             c[j] = 1 / 2 * np.mean((fluctuations[j::] - fluctuations[0:-j]) ** 2)
#
#         plateaux[i] = np.mean(c[int(np.round(eps_max / 3))::])
#         C[i, :] = c
#
#     return C, eps, plateaux
#
# def mean_temporale_auto_correlation(f):
#     mean_C = np.zeros(eps_max)
#     eps_mean_C = np.zeros(eps_max)
#
#     C_c, eps_C_c, plateau_c = temporale_auto_correlation_par_cycle(f)
#
#     for i in range(eps_max):
#         mean_C[i] = np.mean(C_c[:, i])
#         eps_mean_C[i] = np.mean(eps_C_c[:, i])
#
#     plateau_mean_C = np.mean(mean_C[int(np.round(eps_max / 3))::])
#     inv_mean_C = -mean_C + plateau_mean_C
#
#     return mean_C, eps_mean_C, plateau_mean_C, inv_mean_C, C_c, eps_C_c, plateau_c
#
# # ------------------------------------------
# def df_tab(index, df):
#     df_tab = np.zeros_like(index)
#
#     where_df = np.where(index == 1)
#     for i in range(where_df[0].size):
#         df_tab[where_df[0][i], where_df[1][i]] = df[i]
#
#     return df_tab
#
#
# mean_C_f, eps_mean_C_f, plateau_mean_C_f, inv_mean_C_f, C_c_f, eps_C_c_f, plateau_c_f = mean_correlations(signalevent.f)
#
# grid = plot.make_grid(config)
#
# # Define the colors to be used using rainbow map (or any other map)
# colors = plot.make_colors(signal_flu.nbcycle)
#
# savename = 'flu'
# to_save_fig = None
#
# plot.plot_x_y_multiarray(signal_flu.nbcycle, eps_C_c_f, C_c_f,
#                          '\epsilon (s)', 'C(\epsilon)',
#                          'eps{}'.format(savename), 'C_c',
#                          save=to_save_fig, title='correlation vs eps', colors=colors)
#
# plot.plot_y(plateau_c_f, 'cycle', 'plateau',
#             'cycle{}'.format(savename), 'plateau',
#             save=to_save_fig, title='plateaux', label=None, grid=grid)
#
# print('---- valeaur du plateau moyen est {} ----'.format(plateau_mean_C_f))
#
#
# fig, ax = plot.belleFigure('$\epsilon (s)$', '$C(\epsilon)$', nfigure=None)
# ax.plot(eps_mean_C_f, inv_mean_C_f, 'b')
# ax.plot(eps_mean_C_f, np.zeros(eps_max), 'r')
# save = None #to_save_fig + 'inv_mean_C_vs_eps' + savename
# plot.fioritures(ax, fig, title='mean correlation vs eps', label=None, grid=None, save=save)
#
# df = df_tab(signalevent.index_df_tt, signalevent.df_tt)
#
#
# mean_C_df, eps_mean_C_df, plateau_mean_C_df, inv_mean_C_df, C_c_df, eps_C_c_df, plateau_c_df = mean_correlations(df)
#
# grid = plot.make_grid(config)
#
# # Define the colors to be used using rainbow map (or any other map)
# colors = plot.make_colors(signal_flu.nbcycle)
#
# savename = 'flu'
# to_save_fig = None
#
# plot.plot_x_y_multiarray(signal_flu.nbcycle, eps_C_c_df, C_c_df,
#                          '\epsilon (s)', 'C(\epsilon)',
#                          'eps{}'.format(savename), 'C_c',
#                          save=to_save_fig, title='correlation vs eps', colors=colors)
#
# plot.plot_y(plateau_c_df, 'cycle', 'plateau',
#             'cycle{}'.format(savename), 'plateau',
#             save=to_save_fig, title='plateaux', label=None, grid=grid)
#
# print('---- valeaur du plateau moyen est {} ----'.format(plateau_mean_C_df))
#
#
# fig, ax = plot.belleFigure('$\epsilon (s)$', '$C(\epsilon)$', nfigure=None)
# ax.plot(eps_mean_C_df, inv_mean_C_df, 'b')
# ax.plot(eps_mean_C_df, np.zeros(eps_max), 'r')
# save = None #to_save_fig + 'inv_mean_C_vs_eps' + savename
# plot.fioritures(ax, fig, title='mean correlation vs eps', label=None, grid=None, save=save)
#
#
#
#
# where_df_large = np.where(df >= 3e-1)
# df_large = np.zeros_like(df)
# df_large[where_df_large[0], where_df_large[1]] = 1
# mean_Ca_df, eps_mean_Ca_df, plateau_mean_Ca_df, inv_mean_Ca_df, C_c_df, eps_Ca_c_df, plateau_a_c_df = mean_temporale_auto_correlation(df_large)
#
# plot.plot_x_y_multiarray(signal_flu.nbcycle, eps_Ca_c_df, C_c_df,
#                          '\epsilon (s)', 'C(\epsilon)',
#                          'eps{}'.format(savename), 'C_c',
#                          save=to_save_fig, title='correlation vs eps', colors=colors)
#
# plot.plot_y(plateau_c_df, 'cycle', 'plateau',
#             'cycle{}'.format(savename), 'plateau',
#             save=to_save_fig, title='plateaux', label=None, grid=grid)
#
# print('---- valeaur du plateau moyen est {} ----'.format(plateau_mean_Ca_df))
#
#
# fig, ax = plot.belleFigure('$\epsilon (s)$', '$C(\epsilon)$', nfigure=None)
# ax.plot(eps_mean_Ca_df, inv_mean_Ca_df, 'b')
# ax.plot(eps_mean_Ca_df, np.zeros(eps_max), 'r')
# save = None #to_save_fig + 'inv_mean_C_vs_eps' + savename
# plot.fioritures(ax, fig, title='mean correlation vs eps', label=None, grid=None, save=save)
