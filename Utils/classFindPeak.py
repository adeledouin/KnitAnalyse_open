import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


class Derivee():
    # ---------------------------------------------------------#
    def __init__(self, f, ext, t):

        self.f = f
        self.ext = ext
        self.t = t

        self.der_f, self.x_der_f = self.der_t()
        
        self.signe_der_f, self.x_signe_der_f = self.signe_der(self.der_f, self.x_der_f)

        self.der_signe_der_f, self.x_der_signe_der_f = self.der_signe_der(self.signe_der_f, self.x_signe_der_f)

    # ---------------------------------------------------------#
    def der_f(self):
        der_f = np.diff(self.f) #/ np.diff(self.ext)
        x_der_f = self.ext[0:-1]

        return der_f, x_der_f
    
    # ---------------------------------------------------------#
    def der_t(self):
        der_f = np.diff(self.f) / np.diff(self.t)
        x_der_f = self.ext[0:-1]

        return der_f, x_der_f
    # ---------------------------------------------------------#
    def signe_der(self, df, x):
        signe_der_f = np.sign(df)
        x_signe_der_f = x

        return signe_der_f, x_signe_der_f

    # ---------------------------------------------------------#
    def der_signe_der(self, df, x):
        der_signe_der_f = np.diff(df)
        x_der_signe_der_f = x[0:- 2]

        return der_signe_der_f, x_der_signe_der_f

    # ---------------------------------------------------------#
    def plot_derivees(self, plot, df, x_df, sign_df, x_sign_df, dsigne_df, x_dsigne_df):
        ''' vérifie position des max par rapport aux signal initial et ses dérivée
        => sert à débeuguer et vérifier le décalage des indices'''

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(x_df, df, '.')
        plot.plt.ylabel('dérivée de f')
        plot.plt.show()

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(x_sign_df, sign_df, '.')
        plot.plt.ylabel('signe dérivée de f')
        plot.plt.show()

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(x_dsigne_df, dsigne_df, '.')
        plot.plt.ylabel('dérivée du signe de dérivée de f')
        plot.plt.show()

    # ---------------------------------------------------------#
    def find_ones(self, plot):
        where_ones = np.where(self.der_signe_der_f == 1)[0]
        where_minusones = np.where(self.der_signe_der_f == -1)[0]

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(self.x_der_f, self.der_f_ext, 'b.')
        plot.plt.plot(self.x_der_f[1 + where_ones], self.der_f_ext[1 + where_ones], 'r*')
        plot.plt.plot(self.x_der_f[1 + where_minusones], self.der_f_ext[1 + where_minusones], 'm*')
        plot.plt.ylabel('les +-1 dans la dérivée de f')
        plot.plt.show()

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(self.ext, self.f, 'b.')
        plot.plt.plot(self.ext[1 + where_ones], self.f[1 + where_ones], 'r*')
        plot.plt.plot(self.ext[1 + where_minusones], self.f[1 + where_minusones], 'm*')
        plot.plt.ylabel('les +-1 dans signal initial')
        plot.plt.show()

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(self.x_der_signe_der_f, self.der_signe_der_f, 'b.')
        plot.plt.plot(self.x_der_signe_der_f[where_ones], self.der_signe_der_f[where_ones], 'r*')
        plot.plt.plot(self.x_der_signe_der_f[where_minusones], self.der_signe_der_f[where_minusones], 'm*')
        plot.plt.ylabel('les +-1 dans la dérivée du signe de la dérivée de f')
        plot.plt.show()


class DeriveeFiltreGaussien():
    # ---------------------------------------------------------#
    def __init__(self, f, ext, t):
        self.f = f
        self.ext = ext
        self.t = t

        self.der_f, self.x_der_f = self.der_t()

        self.signe_der_f, self.x_signe_der_f = self.signe_der(self.der_f, self.x_der_f)

        self.der_signe_der_f, self.x_der_signe_der_f = self.der_signe_der(self.signe_der_f, self.x_signe_der_f)

    # ---------------------------------------------------------#
    def der_f(self):
        der_f = np.diff(self.f)  # / np.diff(self.ext)
        x_der_f = self.ext[0:-1]

        return der_f, x_der_f

    # ---------------------------------------------------------#
    def der_t(self):
        der_f = np.diff(self.f) / np.diff(self.t)
        der_f = gaussian_filter1d(der_f, 0.1)
        x_der_f = self.ext[0:-1]

        return der_f, x_der_f

    # ---------------------------------------------------------#
    def signe_der(self, df, x):
        signe_der_f = np.sign(df)
        x_signe_der_f = x

        return signe_der_f, x_signe_der_f

    # ---------------------------------------------------------#
    def der_signe_der(self, df, x):
        der_signe_der_f = np.diff(df)
        x_der_signe_der_f = x[0:- 2]

        return der_signe_der_f, x_der_signe_der_f

    # ---------------------------------------------------------#
    def plot_derivees(self, plot, df, x_df, sign_df, x_sign_df, dsigne_df, x_dsigne_df):
        ''' vérifie position des max par rapport aux signal initial et ses dérivée
        => sert à débeuguer et vérifier le décalage des indices'''

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(x_df, df, '.')
        plot.plt.ylabel('dérivée de f')
        plot.plt.show()

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(x_sign_df, sign_df, '.')
        plot.plt.ylabel('signe dérivée de f')
        plot.plt.show()

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(x_dsigne_df, dsigne_df, '.')
        plot.plt.ylabel('dérivée du signe de dérivée de f')
        plot.plt.show()

    # ---------------------------------------------------------#
    def find_ones(self, plot):
        where_ones = np.where(self.der_signe_der_f == 1)[0]
        where_minusones = np.where(self.der_signe_der_f == -1)[0]

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(self.x_der_f, self.der_f_ext, 'b.')
        plot.plt.plot(self.x_der_f[1 + where_ones], self.der_f_ext[1 + where_ones], 'r*')
        plot.plt.plot(self.x_der_f[1 + where_minusones], self.der_f_ext[1 + where_minusones], 'm*')
        plot.plt.ylabel('les +-1 dans la dérivée de f')
        plot.plt.show()

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(self.ext, self.f, 'b.')
        plot.plt.plot(self.ext[1 + where_ones], self.f[1 + where_ones], 'r*')
        plot.plt.plot(self.ext[1 + where_minusones], self.f[1 + where_minusones], 'm*')
        plot.plt.ylabel('les +-1 dans signal initial')
        plot.plt.show()

        plot.plt.figure(figsize=(15, 5))
        plot.plt.subplot(121)
        plot.plt.plot(self.x_der_signe_der_f, self.der_signe_der_f, 'b.')
        plot.plt.plot(self.x_der_signe_der_f[where_ones], self.der_signe_der_f[where_ones], 'r*')
        plot.plt.plot(self.x_der_signe_der_f[where_minusones], self.der_signe_der_f[where_minusones], 'm*')
        plot.plt.ylabel('les +-1 dans la dérivée du signe de la dérivée de f')
        plot.plt.show()


# ------------------------------------------------------------------#
# ------------------------------------------------------------------#

class FindPeak():
    ''' trouves les min et max d'un signal '''

    # ---------------------------------------------------------#
    def __init__(self, signal, cycle, brut_signal=False):

        self.signal = signal
        self.cycle = cycle
        self.brut_signal = brut_signal

    # ---------------------------------------------------------#
    def clear_ones(self):
        work_on = self.signal

        # plt.plot(work_on, 'b.')
        # plt.show()

        for j in range(np.size(work_on)):
            # regarde les up
            if j == np.size(work_on) - 1:
                if work_on[j] == -1:
                    work_on[j] = 0
                if work_on[j] == 1:
                    work_on[j] = 2

            elif work_on[j] == -1:
                # print('le j = {} est un -1'.format(j))
                k = j + 1  # a voir peut etre j+2
                # il y a un up tout à droite
                if work_on[k] == -1:
                    # print('le k = {} est un -1, 1 pt apres j'.format(k))
                    work_on[j] = 0
                    work_on[k] = -2
                # pas de up juste un plateau
                elif work_on[k] == 1:
                    # print('le k = {} est un 1, 1 pt apres j'.format(k))
                    work_on[j] = 0
                    work_on[k] = 0
                # il faut regarder plus loin - cas plateau de plusieurs points
                elif work_on[k] == 0:
                    pt = 1
                    while work_on[k] == 0:
                        k = k + 1
                        pt = pt + 1
                    # il y a un up à la fin du plateau
                    if work_on[k] == -1:
                        # print('le k = {} est un -1, {} pt apres j'.format(k, pt))
                        work_on[j] = 0
                        work_on[k] = -2
                    # pas de up
                    elif work_on[k] == 1:
                        # print('le k = {} est un -1, {} pt apres j'.format(k, pt))
                        work_on[j] = 0
                        work_on[k] = 0

            # regarde les down
            elif work_on[j] == 1:
                # print('le j = {} est un 1'.format(j))
                k = j + 1  # a voir peut etre j+2
                # pas de down juste un plateau
                if work_on[k] == -1:
                    work_on[j] = 0
                    work_on[k] = 0
                # il y a un down tout à gauche
                elif work_on[k] == 1:
                    work_on[j] = 2
                    work_on[k] = 0
                # il faut regarder plus loin - cas plateau de plusieurs points
                elif work_on[k] == 0:
                    while work_on[k] == 0:
                        k = k + 1
                    # il y a un down à la fin du plateau
                    if work_on[k] == -1:
                        work_on[j] = 0
                        work_on[k] = 0
                    # pas de up
                    elif work_on[k] == 1:
                        work_on[j] = 2
                        work_on[k] = 0

        # plt.plot(work_on, 'r.')
        # plt.show()

        return work_on

    # ---------------------------------------------------------#
    def clear_bord(self, min_indice, max_indice, min_indice_size, max_indice_size):

        # min en premier
        if min_indice[0] < max_indice[0]:
            print('%d avec min en premier' % (self.cycle))
            min_indice = min_indice[1::]
            min_indice_size = min_indice_size - 1

        # max en derier
        if max_indice[-1] > min_indice[-1]:
            print('%d avec max en dernier' % (self.cycle))
            max_indice = max_indice[0:-1]
            max_indice_size = max_indice_size - 1

        return min_indice, max_indice, min_indice_size, max_indice_size

    # ---------------------------------------------------------#
    def recup_min_max_indices(self):

        if self.brut_signal == True:

            max_indice = np.where((self.signal == -2) | (self.signal == -1))[0]
            min_indice = np.where((self.signal == 2) | (self.signal == 1))[0]
            max_indice_size = np.size(max_indice)
            min_indice_size = np.size(min_indice)

        else:
            good_f = self.clear_ones()

            max_indice = np.where(good_f == -2)[0]
            min_indice = np.where(good_f == 2)[0]
            max_indice_size = np.size(max_indice)
            min_indice_size = np.size(min_indice)

            print(max_indice_size, min_indice_size)
            # print('avant clean bord',max_indice_size,min_indice_size)
            min_indice, max_indice, min_indice_size, max_indice_size = self.clear_bord(min_indice, max_indice,
                                                                                       min_indice_size, max_indice_size)

            # print('apres clean bord', max_indice_size, min_indice_size)
        return min_indice, max_indice, min_indice_size, max_indice_size
