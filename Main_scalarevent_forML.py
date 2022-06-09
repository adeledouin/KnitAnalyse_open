# %%
import numpy as np
import timeit

from classConfig import Config
from Datas.classSignal import SignalForce, SignalImg, VariationsScalar
from Datas.classEvent import ForceEvent
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
NN_data = ''
Sm = False
display_figure = True
display_figure_reg = False
display_figure_variations = False

# %% ################### Partie 0 : Detect events ##################################
print('#-------------detect events--------------#')

signal_flu = SignalForce(config, signaltype, NN_data)

if config.img:
    signal_img = SignalImg(config, signaltype, NN_data, fields=False)

signalevent = ForceEvent(config, signal_flu.f, signal_flu.ext, signal_flu.t, signaltype, NN_data, Sm=Sm, display_figure_debug=False)
df_tab = signalevent.df_tab()

decade_grid = np.array([5e-3, 3e-2, 3e-1, 3e0,  3e1])
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


# %% #####  t btw events

major = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
decade = np.array([0, 5e-3, 3e-2, 3e-1, 3e0, 3e1])
decade_grid = np.array([3e-3, 3e-2, 3e-1, 3e0,  3e1])
log_classes = np.array([1e-4, 3e-3, 3e-2, 3e-1, 3e0, 3e1])
class_grid = log_classes
exposants = [4, 3, 2, 1, 0]
save_seuil = ['0', '3_3', '3_2', '3_0']

print('#-------------sub signal : df--------------#')
start_time = timeit.default_timer()

df_tab = signalevent.df_tab()

hist, _ = np.histogram(df_tab.reshape(df_tab.size), bins=decade, density=False)
print('prop_on_decade in df in all train = {}'.format(np.round(hist / df_tab.size * 100, 2)))

x_Pdf_dt_d = [0 for i in range(decade.size-1)]
y_Pdf_dt_d = [0 for i in range(decade.size-1)]

x_Pdf_nbp_d = [0 for i in range(decade.size-1)]
y_Pdf_nbp_d = [0 for i in range(decade.size-1)]

for i in range(decade.size-2):
    df_seuil, dt_seuil, index_df_seuil, number_df_seuil = signalevent.df_seuil(decade[i], decade[i+1], df_tab)

    dt_btw, index_dt_btw, number_dt_btw = signalevent.time_btw_df(index_df_seuil, number_df_seuil)
    nb_pts = np.round(dt_btw/0.04, 0)
    # if i == 0:
    #     add = signalevent.f.shape[0]*signalevent.f.shape[1]-signalevent.dt_tt.size
    #     dt_btw = np.concatenate((dt_btw, np.ones(add)*0.04))
    #     nb_pts = np.concatenate((nb_pts, np.ones(add)))

    y_Pdf_dt_d[i], x_Pdf_dt_d[i] = plot.histo.my_histo(dt_btw, np.min(dt_btw),
                                       np.max(dt_btw),
                                       'log', 'log', density=2, binwidth=None, nbbin=100)

    y_Pdf_nbp_d[i], x_Pdf_nbp_d[i] = plot.histo.my_histo(nb_pts, np.min(nb_pts),
                                       np.max(nb_pts),
                                       'log', 'log', density=2, binwidth=1.6/0.04, nbbin=None)

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

yname = '\delta t_{btw}(s)'
ysave = 'dt_btw'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
for i in range(decade.size-1):
    ax.plot(x_Pdf_dt_d[i], y_Pdf_dt_d[i], '.', label='decade {}'.format(i))
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None  # path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save, major=None)

plot.plot_reg(dt_btw, np.min(dt_btw), np.max(dt_btw),
              '\Delta \delta tbtw', 'Pdf(\Delta \delta tbtw)',
              'pdf', 'delta_btw',
              minreg=-0.7,
              maxreg=0.3,
              save=signalevent.to_save_fig,
              powerlaw=None)

# yname = 'nb pts btw'
# ysave = 'dt_btw'
# title = True
# label = True
# grid = decade_grid
# fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
# for i in range(decade.size-1):
#     ax.plot(x_Pdf_nbp_d[i], y_Pdf_nbp_d[i], '.', label='decade {}'.format(i))
# plot.plt.xscale('log')
# plot.plt.yscale('log')
# save = None  # path_Tex + '{}'.format('pdf_df_all')
# plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save, major=None)


i = 4
df_seuil, dt_seuil, index_df_seuil, number_df_seuil = signalevent.df_seuil(decade[i], decade[i + 1], df_tab)

dt_btw, index_dt_btw, number_dt_btw = signalevent.time_btw_df(index_df_seuil, number_df_seuil)
nb_pts = np.round(dt_btw /0.04, 0)

y_Pdf_dt_d[i], x_Pdf_dt_d[i] = plot.histo.my_histo(dt_btw, np.min(dt_btw),
                                   np.max(dt_btw),
                                   'log', 'log', density=2, binwidth=None, nbbin=100)

y_Pdf_nbp_d[i], x_Pdf_nbp_d[i] = plot.histo.my_histo(nb_pts, np.min(nb_pts),
                                   np.max(nb_pts),
                                   'log', 'log', density=2, binwidth=40/0.04, nbbin=None)

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
yname = '\delta t_{btw}(s)'
ysave = 'dt_btw'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
for i in range(decade.size-1):
    ax.plot(x_Pdf_dt_d[i], y_Pdf_dt_d[i], '.', label='decade {}'.format(i))
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None  # path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save, major=None)

plot.plot_reg(dt_btw, np.min(dt_btw), np.max(dt_btw),
              '\Delta \delta tbtw', 'Pdf(\Delta \delta tbtw)',
              'pdf', 'delta_btw',
              minreg=None,
              maxreg=None,
              save=signalevent.to_save_fig,
              powerlaw=None)

# yname = 'nb pts btw'
# ysave = 'dt_btw'
# title = True
# label = True
# grid = decade_grid
# fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
# for i in range(decade.size-1):
#     ax.plot(x_Pdf_nbp_d[i], y_Pdf_nbp_d[i], '.', label='decade {}'.format(i))
# plot.plt.xscale('log')
# plot.plt.yscale('log')
# save = None  # path_Tex + '{}'.format('pdf_df_all')
# plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save, major=None)

yname = '\delta t_{btw}(s)'
ysave = 'dt_btw'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(dt_btw, np.min(dt_btw),
                                   np.max(dt_btw),
                                   'log', 'log', density=2, binwidth=None, nbbin=150)
ax.plot(x_Pdf, y_Pdf, '.', label='decade {}'.format(i))
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None  # path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save, major=None)


# %% ################## Partie 3 : sum df btwpict ##################################
nb_df_tt_btw = signalevent.nb_df_btwpict(df_tab, signal_img.index_picture)
sum_df_tt_btw_tab, sum_df_tt_btw, max_df_tt_btw_tab, max_df_tt_btw, index_df_tt_btw, number_df_tt_btw = signalevent.df_btw_pict(
    df_tab,
    signal_img.index_picture,
    signal_img.numbertot_picture,
    nb_df_tt_btw)


if config.mix_set:
    grid = plot.make_grid(config)
else:
    grid = None

colors = plot.make_colors(signal_flu.nbcycle)

# %% ##### sum_df
yname = '\Sigma(\delta f)'
ysave = 'df'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(sum_df_tt_btw, 1e-4,
                                   np.max(sum_df_tt_btw),
                                   'log', 'log', density=2, binwidth=None, nbbin=70)

ax.plot(x_Pdf, y_Pdf, '.')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None #path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=None, grid=decade_grid, save=save, major=None)

# %% ##### max_df
yname = 'Max(\delta f)'
ysave = 'df'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
y_Pdf, x_Pdf = plot.histo.my_histo(max_df_tt_btw, 1e-4,
                                   np.max(max_df_tt_btw),
                                   'log', 'log', density=2, binwidth=None, nbbin=70)

ax.plot(x_Pdf, y_Pdf, '.')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None #path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=None, grid=decade_grid, save=save, major=None)