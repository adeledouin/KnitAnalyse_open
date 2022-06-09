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

if display_figure:
    i = config_plot['signalevent_cycle_plot']

    plot.plot_x_y(signalevent.ext[i, :], signalevent.f[i, :], 'L_{w} (mm)', '\delta f',
                  'Lw_c{}'.format(i), signaltype, save=signalevent.to_save_fig, title='Knit-quakes noise', pts='-')

    ini = config_plot['signalevent_training_event_i']
    fin = config_plot['signalevent_training_event_f']

    plot.plot_x_y(signalevent.t[i, ini:fin], signalevent.f[i, ini:fin], 't (s)', '\delta f',
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

variations_df_tt = VariationsScalar(config, pourcentage=5, f=signalevent.df_tt, ext=signal_flu.ext, t=signal_flu.t,
                                    index=signalevent.index_df_tt, number=signalevent.number_df_tt, directsignal=False,
                                    signaltype=signaltype, NN_data=NN_data, ftype='force',
                                    fname='df' + signalevent.savename_df_tt,
                                    stats=display_figure_variations)

variations_dt_tt = VariationsScalar(config, pourcentage=5, f=signalevent.dt_tt, ext=signal_flu.ext, t=signal_flu.t,
                                    index=signalevent.index_df_tt, number=signalevent.number_df_tt, directsignal=False,
                                    signaltype=signaltype, NN_data=NN_data, ftype='force', fname='dt_tt',
                                    stats=display_figure_variations)

if display_figure:
    plot.Pdf_loglog(signalevent.df_tt, variations_df_tt.stats_f.min, variations_df_tt.stats_f.max,
                    '\Delta \delta f', 'df' + signalevent.savename_df_tt, save=signalevent.to_save_fig, nbbin=150)

    plot.Pdf_linlog(signalevent.df_tt, variations_df_tt.stats_f.min, variations_df_tt.stats_f.max,
                    '\Delta \delta f', 'df' + signalevent.savename_df_tt + 'linlog', save=signalevent.to_save_fig,
                    nbbin=150)

# %% ################## Partie 1 : find seuil_bruit ##################################
print('#-------------trouver le seuil--------------#')

exposants, seuils, save_seuils, nb_seuils, which_seuil, nbclasses = config.config_scalarevent(Sm)

which_seuil = 4
X_df = [0 for i in range(nb_seuils)]
Y_df = [0 for i in range(nb_seuils)]
X_dt = [0 for i in range(nb_seuils)]
Y_dt = [0 for i in range(nb_seuils)]

for i in range(nb_seuils):
    exposant = exposants[i]
    seuil = seuils[i]

    Y_df[i], X_df[i] = histo.my_histo(signalevent.df_tt, seuil, variations_df_tt.stats_f.max, 'log',
                                      'log', density=2, binwidth=None, nbbin=100)

fig, ax = plot.belleFigure('$\Delta \delta f$', '$P(\Delta \delta f)$', nfigure=None)
for i in range(nb_seuils):
    ax.plot(X_df[i], Y_df[i], '.',
            label='seuil =' + np.str(seuils[i] * 10 ** exposants[i]) + 'e-' + np.str(exposants[i]) + 'N')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = signalevent.to_save_fig + 'pdf_' + 'df' + signalevent.savename_df_tt + '_seuils' + '.pdf'
plot.fioritures(ax, fig, title='pdf de df pour differents seuils', label=True, grid=None, save=save)

# %% ################# Partie 2 : Stats des events sup seuil ##################################

print('#-------------df tt good seuil--------------#')

if display_figure:
    print('#-------------df tt--------------#')

    plot_pdf_event(config_plot, plot, signalevent, signalevent.df_tt, variations_df_tt, signalevent.savename_df_tt,
                   seuils[which_seuil], exposants[which_seuil], display_figure_reg)

    if display_figure_variations:
        plot_variations_df_tt(plot, signalevent, variations_df_tt, which_seuil, seuils, exposants, display_figure_reg)

exposant = exposants[which_seuil]
seuil = seuils[which_seuil]
save_seuil = save_seuils[which_seuil]

# %% ##### df seuil
print('#-------------sub signal : df--------------#')
start_time = timeit.default_timer()
df_tab = signalevent.df_tab()

df_seuil, dt_seuil, index_df_seuil, number_df_seuil = signalevent.df_seuil(seuil, None, df_tab)

df_seuil_tab = signalevent.df_seuil_tab(df_seuil, index_df_seuil)

variations_df_seuil = VariationsScalar(config, pourcentage=5, f=df_seuil, ext=signal_flu.ext, t=signal_flu.t,
                                       index=index_df_seuil, number=number_df_seuil, directsignal=False,
                                       signaltype=signaltype, NN_data=NN_data, ftype='force',
                                       fname='df' + signalevent.savename_df_seuil)

if display_figure:
    yname = '\delta f'
    ysave = 'df'
    title = True
    label = True
    grid = None
    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
    y_Pdf, x_Pdf = plot.histo.my_histo(df_seuil, np.min(df_seuil),
                                       np.max(df_seuil),
                                       'log', 'log', density=2, binwidth=None, nbbin=70)

    ax.plot(x_Pdf, y_Pdf, '.')
    plot.plt.xscale('log')
    plot.plt.yscale('log')
    save = signalevent.to_save_fig + 'pdf_' + ysave + signalevent.savename_df_tt
    plot.fioritures(ax, fig, title=None, label=None, grid=grid, save=save, major=None)

    plot.plot_reg(df_seuil, seuil, variations_df_seuil.stats_f.max,
                  '\Delta \delta f', 'Pdf(\Delta \delta f)',
                  'pdf', 'df' + 'seuil',
                  minreg=-1.5,
                  maxreg=0.5,
                  save=signalevent.to_save_fig,
                  label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(
                      exposant) + 'N',
                  powerlaw=-1.5)


stop_time = timeit.default_timer()
print('tps pour calculer powerlaw :', stop_time - start_time)

# %% ##### dt seuil
print('#-------------sub signal : dt--------------#')

start_time = timeit.default_timer()

variations_dt_seuil = VariationsScalar(config, pourcentage=5, f=dt_seuil, ext=signal_flu.ext, t=signal_flu.t,
                                       index=index_df_seuil, number=number_df_seuil, directsignal=False,
                                       signaltype=signaltype, NN_data=NN_data, ftype='force', fname='dt_seuil')

if display_figure:

    yname = '\delta t'
    ysave = 'dt'
    title = True
    label = True
    grid = None
    fig, ax = plot.belleFigure('$log10({})$'.format(yname), '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
    y_Pdf, x_Pdf = plot.histo.my_histo(dt_seuil, np.min(dt_seuil),
                                       np.max(dt_seuil),
                                       'lin', 'log', density=2, binwidth=None, nbbin=70)

    ax.plot(x_Pdf, y_Pdf, '.')
    # plot.plt.xscale('log')
    plot.plt.yscale('log')
    save = signalevent.to_save_fig + 'pdf_' + ysave + signalevent.savename_df_tt
    plot.fioritures(ax, fig, title=None, label=None, grid=grid, save=save, major=None)

    plot.Pdf_linlog(dt_seuil, seuil, variations_dt_seuil.stats_f.max,
                    '\Delta t', 'dt_seuil',
                    save=signalevent.to_save_fig,
                    label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(exposant) + 'N')

    plot.Pdf_linlog(dt_seuil, seuil, variations_dt_seuil.stats_f.max,
                    '\Delta t', 'dt_seuil',
                    save=signalevent.to_save_fig,
                    label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(exposant) + 'N',
                    defreg=True)

    if display_figure_reg:
        plot.plot_reg(dt_seuil, seuil, variations_dt_seuil.stats_f.max,
                      '\Delta \delta f', 'Pdf(\Delta \delta f)',
                      'pdf', 'dt_seuil',
                      minreg=config_plot[
                          'signalevent_pdf(dt_seuil)_minreg'],
                      maxreg=config_plot[
                          'signalevent_pdf(dt_seuil)_maxreg'],
                      xaxis='lin',
                      save=signalevent.to_save_fig,
                      label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(
                          exposant) + 'N',
                      powerlaw=None)


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
              label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(
                  exposant) + 'N',
              powerlaw=None)

yname = 'nb pts btw'
ysave = 'dt_btw'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
for i in range(decade.size-1):
    ax.plot(x_Pdf_nbp_d[i], y_Pdf_nbp_d[i], '.', label='decade {}'.format(i))
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None  # path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save, major=None)


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
              label='seuil =' + np.str(seuil * 10 ** exposant) + 'e-' + np.str(
                  exposant) + 'N',
              powerlaw=None)

yname = 'nb pts btw'
ysave = 'dt_btw'
title = True
label = True
grid = decade_grid
fig, ax = plot.belleFigure('${}$'.format(yname), '${}({})$'.format('Count_{log10}', yname), nfigure=None)
for i in range(decade.size-1):
    ax.plot(x_Pdf_nbp_d[i], y_Pdf_nbp_d[i], '.', label='decade {}'.format(i))
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None  # path_Tex + '{}'.format('pdf_df_all')
plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save, major=None)

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

# # %% ################## Partie 3 : sum df btwpict ##################################
if config.img:
    if not config.mix_set:
        nb_df_tt_btw = signalevent.nb_df_btwpict(df_tab, signal_img.index_picture)
        sum_df_tt_btw_tab, sum_df_tt_btw, max_df_tt_btw_tab, max_df_tt_btw, index_df_tt_btw, number_df_tt_btw = signalevent.df_btw_pict(
            df_tab,
            signal_img.index_picture,
            signal_img.number_picture,
            nb_df_tt_btw)

        nb_df_seuil_btw = signalevent.nb_df_btwpict(df_seuil_tab, signal_img.index_picture)
        sum_df_seuil_btw_tab, sum_df_seuil_btw, max_df_seuil_btw_tab, max_df_seuil_btw, index_df_seuil_btw, number_df_seuil_btw = signalevent.df_btw_pict(
            df_seuil_tab,
            signal_img.index_picture,
            signal_img.number_picture,
            nb_df_seuil_btw)
    else:
        nb_df_tt_btw = signalevent.nb_df_btwpict(df_tab, signal_img.index_picture)
        sum_df_tt_btw_tab, sum_df_tt_btw, max_df_tt_btw_tab, max_df_tt_btw, index_df_tt_btw, number_df_tt_btw = signalevent.df_btw_pict(
            df_tab,
            signal_img.index_picture,
            signal_img.numbertot_picture,
            nb_df_tt_btw)

        nb_df_seuil_btw = signalevent.nb_df_btwpict(df_seuil_tab, signal_img.index_picture)
        sum_df_seuil_btw_tab, sum_df_seuil_btw, max_df_seuil_btw_tab, max_df_seuil_btw, index_df_seuil_btw, number_df_seuil_btw = signalevent.df_btw_pict(
            df_seuil_tab,
            signal_img.index_picture,
            signal_img.numbertot_picture,
            nb_df_seuil_btw)

    variations_nb_df_tt_btw = VariationsScalar(config, pourcentage=5, f=nb_df_tt_btw, ext=signal_flu.ext, t=signal_flu.t,
                                           index=signal_img.index_picture, number=signal_img.numbertot_picture,
                                           directsignal=False,
                                           signaltype=signaltype, NN_data=NN_data, ftype='img', fname='nb_df_tt_btw', stats=True)

    variations_nb_df_seuil_btw = VariationsScalar(config, pourcentage=5, f=nb_df_seuil_btw, ext=signal_flu.ext,
                                              t=signal_flu.t,
                                              index=signal_img.index_picture, number=signal_img.numbertot_picture,
                                              directsignal=False,
                                              signaltype=signaltype, NN_data=NN_data, ftype='img', fname='nb_df_seuil_btw', stats=True)

    variations_sum_df_tt_btw = VariationsScalar(config, pourcentage=5, f=sum_df_tt_btw, ext=signal_flu.ext, t=signal_flu.t,
                                            index=index_df_tt_btw, number=number_df_tt_btw, directsignal=False,
                                            signaltype=signaltype, NN_data=NN_data, ftype='img', fname='sum_df_tt_btw', stats=True)

    variations_sum_df_seuil_btw = VariationsScalar(config, pourcentage=5, f=sum_df_seuil_btw, ext=signal_flu.ext,
                                               t=signal_flu.t,
                                               index=index_df_tt_btw, number=number_df_seuil_btw,
                                               directsignal=False,
                                               signaltype=signaltype, NN_data=NN_data, ftype='img', fname='sum_df_seuil_btw', stats=True)

    variations_max_df_tt_btw = VariationsScalar(config, pourcentage=5, f=max_df_tt_btw, ext=signal_flu.ext, t=signal_flu.t,
                                            index=index_df_tt_btw, number=number_df_tt_btw, directsignal=False,
                                            signaltype=signaltype, NN_data=NN_data, ftype='img', fname='max_df_tt_btw', stats=True)

    variations_max_df_seuil_btw = VariationsScalar(config, pourcentage=5, f=max_df_seuil_btw, ext=signal_flu.ext,
                                               t=signal_flu.t,
                                               index=index_df_tt_btw, number=number_df_seuil_btw,
                                               directsignal=False,
                                               signaltype=signaltype, NN_data=NN_data, ftype='img', fname='max_df_seuil_btw', stats=True)

    if config.mix_set:
        grid = plot.make_grid(config)
    else:
        grid = None

    colors = plot.make_colors(signal_flu.nbcycle)

    # %% ##### nb df
    if display_figure:
        fig, ax = plot.belleFigure('img', '$N(\Delta \delta f.img^{-1})$', nfigure=None)
        ax.plot(np.arange(np.size(nb_df_tt_btw)), nb_df_tt_btw, '.',
                label='$<N>= {}$'.format(np.round(variations_nb_df_tt_btw.stats_f.mean)))
        ax.plot(np.arange(np.size(nb_df_seuil_btw)), nb_df_seuil_btw, '.',
                label='$seuil = {}$, $<N>= {}$'.format(seuil,
                                                       np.round(variations_nb_df_seuil_btw.stats_f.mean)))
        save = signal_img.to_save_fig + 'nb_df_btwpict' + '.pdf'
        plot.fioritures(ax, fig, title='nb chutes btw pict', label=True, grid=None,save=save)

        plot_pdf_event_comparaison(plot, nb_df_tt_btw, nb_df_seuil_btw, variations_nb_df_tt_btw, variations_nb_df_seuil_btw,
                                   'N(\Delta \delta f.img^{-1})', 'nb_df_btwpict',
                                   seuil, exposant,
                                   xaxis='lin', yaxis='log', save=signal_img.to_save_fig)


        plot_variations_df_img(plot, signal_img, signalevent, variations_nb_df_tt_btw, variations_nb_df_seuil_btw,
                               'Ndf_{btwpict}', 'nb_df_btwpict', seuil, exposant, save=signal_img.to_save_fig)

# %% ##### sum_df
    if display_figure:

        plot_pdf_event_comparaison(plot, sum_df_tt_btw, sum_df_seuil_btw, variations_sum_df_tt_btw, variations_sum_df_seuil_btw,
                                   '\Sigma_{btwpict}\Delta \delta f', 'sum_df_btwpict',
                                   seuil, exposant,
                                   xaxis='lin', yaxis='log', save=signal_img.to_save_fig)


        plot_variations_df_img(plot, signal_img, signalevent, variations_sum_df_tt_btw, variations_sum_df_seuil_btw,
                               '\Sigma \delta f_{btwpict}', 'sum_df_btwpict', seuil, exposant, save=signal_img.to_save_fig)


    # %% ##### max_df
    if display_figure:

        plot_pdf_event_comparaison(plot, max_df_tt_btw, max_df_seuil_btw, variations_max_df_tt_btw, variations_max_df_seuil_btw,
                                   'Max_{btwpict}(\Delta \delta f)', 'max_df_btwpict',
                                   seuil, exposant,
                                   xaxis='lin', yaxis='log', save=signal_img.to_save_fig)

        plot_variations_df_img(plot, signal_img, signalevent, variations_max_df_tt_btw, variations_max_df_seuil_btw,
                               'Max( \delta f_{btwpict})','max_df_btwpict', seuil, exposant, save=signal_img.to_save_fig)

