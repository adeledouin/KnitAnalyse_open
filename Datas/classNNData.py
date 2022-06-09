import numpy as np

from pathlib import Path

from Utils.classCell import Cell

# ------------------------------------------
class CreateNNData():
    """
    Classe qui permet de séparer les données en train, val et test set.

    Attributes:
        config (class) : config associée à la l'analyse

        signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force
        fname (str) : nom du signal

        to_save (str) : chemin pour save les split

        nbcycle (int) : nombre de cycles dans le signal
        sub_cycles (list[liste]) : liste des cycles par set comptés sur nombre total de cycles dans l'analyse

        train_cycles (list(array)) : cycles appartenant au train set, compté sur le nombre total de cycles
        val_cycles (list(array)) : cycles appartenant au val set, compté sur le nombre total de cycles
        test_cycles (array) : cycles appartenant au test set, compté sur le nombre total de cycles
        train_sub_cycles_NN (list(array) : liste par set des cycles apparetant au train set - compté sur nombre de cycles dans le set
        val_sub_cycles_NN (list(array) : liste par set des cycles apparetant au val set - compté sur nombre de cycles dans le set
        test_sub_cycles_NN (list(array) : liste par set des cycles apparetant au test set - compté sur nombre de cycles dans le set
        train_sub_cycles (list(array) : liste par set des cycles apparetant au train set - compté sur nombre de cycles dans train set
        val_sub_cycles (list(array) : liste par set des cycles apparetant au val set - compté sur nombre de cycles dans val set
        test_sub_cycles (list(array) : liste par set des cycles apparetant au test set - compté sur nombre de cycles dans test set

    """

    # ---------------------------------------------------------#
    def __init__(self, config, signaltype, set_to_keep):
        """
        The constructor for CreateNNData.

        Parameters:
            config (class) : config associée à la l'analyse

            signaltype (str): 'flu', 'flu_rsc', 'rsc_flu_rsc', 'sequence' type du signal en force

        """

        ## Config
        self.config = config

        self.signaltype = signaltype
        self.fname = signaltype

        self.to_save = self.config.global_path_save + self.signaltype + '_NN' + '/'

        if not self.config.mix_set:
            self.nbcycle = self.config.nbcycle
        else:
            self.nbcycle = np.sum(config.nbcycle if set_to_keep is None else config.nbcycle[set_to_keep])

            self.sub_cycles = config.sub_cycles if set_to_keep is None else [config.sub_cycles[i] for i in set_to_keep]

        self.train_cycle, self.val_cycle, self.test_cycle = self.get_NN_cycles()

        if self.config.mix_set:
            self.train_sub_cycle_NN, self.val_sub_cycle_NN, self.test_sub_cycle_NN = self.get_sub_cycles_NN()
            self.train_NN_sub_cycle, self.val_NN_sub_cycle, self.test_NN_sub_cycle = self.get_NN_sub_cycles()


    # ------------------------------------------
    def get_NN_cycles(self):

        ## regarde si le fichier existe dejà :
        fileName = self.to_save + self.signaltype + '_train_cycles' + '.npy'
        fileObj = Path(fileName)

        is_fileObj = fileObj.is_file()

        tosave_train_cycle = self.to_save + self.signaltype + '_train_cycles'
        tosave_val_cycle = self.to_save + self.signaltype + '_val_cycles'
        tosave_test_cycle = self.to_save + self.signaltype + '_test_cycles'

        if is_fileObj:
            print('NN cycles déjà enregisté dans {}'.format(fileObj))
            train_cycle = np.load(tosave_train_cycle + '.npy')
            val_cycle = np.load(tosave_val_cycle + '.npy')
            test_cycle = np.load(tosave_test_cycle + '.npy')

            train_cycle = np.sort(train_cycle)
            val_cycle = np.sort(val_cycle)
            test_cycle = np.sort(test_cycle)

        else:
            if not self.config.mix_set:
                train_size = int(np.round(80 / 100 * self.nbcycle))
                val_size = int(np.round(10 / 100 * self.nbcycle))
                test_size = self.nbcycle - train_size - val_size

                p = np.random.permutation(self.nbcycle)

                train_cycle = np.sort(p[0:train_size])
                val_cycle = np.sort(p[train_size:train_size + val_size])
                test_cycle = np.sort(p[train_size + val_size::])
            else:
                train_size = int(np.round(80 / 100 * np.sum(self.nbcycle)))
                val_size = int(np.round(10 / 100 * np.sum(self.nbcycle)))
                test_size = self.nbcycle - train_size - val_size

                p = np.random.permutation(np.arange(self.sub_cycles[0][0], self.sub_cycles[-1][-1]))  #np.sum(self.nbcycle))

                # print('verif random pick des cycles :  nbcycles tot = {}, traille p = {}, p = {}'.format(np.sum(self.nbcycle), p.size, p))

                train_cycle = np.sort(p[0:train_size])
                val_cycle = np.sort(p[train_size:train_size + val_size])
                test_cycle = np.sort(p[train_size + val_size::])

        return train_cycle, val_cycle, test_cycle

    # ------------------------------------------
    def get_NN_sub_cycles(self):

        ## regarde si le fichier existe dejà :
        fileName = self.to_save + self.signaltype + '_train_NN_sub_cycles_1' + '.npy'
        fileObj = Path(fileName)

        is_fileObj = fileObj.is_file()

        tosave_train_cycle = self.to_save + self.signaltype + '_train_NN_sub_cycles'
        tosave_val_cycle = self.to_save + self.signaltype + '_val_NN_sub_cycles'
        tosave_test_cycle = self.to_save + self.signaltype + '_test_NN_sub_cycles'

        if is_fileObj:
            print('NN cycles déjà enregisté')
            recup_train = Cell(tosave_train_cycle, self.config.nb_set)
            recup_val = Cell(tosave_val_cycle, self.config.nb_set)
            recup_test = Cell(tosave_test_cycle, self.config.nb_set)

            train_sub_cycle = recup_train.reco_cell()
            val_sub_cycle = recup_val.reco_cell()
            test_sub_cycle = recup_test.reco_cell()

        else:
            train_sub_cycle = [0 for i in range(self.config.nb_set)]
            val_sub_cycle = [0 for i in range(self.config.nb_set)]
            test_sub_cycle = [0 for i in range(self.config.nb_set)]

            nb_cycles_train = 0
            nb_cycles_val = 0
            nb_cycles_test = 0
            for i in range(self.config.nb_set):
                train_sub_cycle[i] = np.arange(nb_cycles_train, nb_cycles_train + np.size(self.train_sub_cycle_NN[i]))
                val_sub_cycle[i] = np.arange(nb_cycles_val, nb_cycles_val + np.size(self.val_sub_cycle_NN[i]))
                test_sub_cycle[i] = np.arange(nb_cycles_test, nb_cycles_test + np.size(self.test_sub_cycle_NN[i]))
                nb_cycles_train = nb_cycles_train + np.size(self.train_sub_cycle_NN[i])
                nb_cycles_val = nb_cycles_val + np.size(self.val_sub_cycle_NN[i])
                nb_cycles_test = nb_cycles_test + np.size(self.test_sub_cycle_NN[i])

        return train_sub_cycle, val_sub_cycle, test_sub_cycle

    # ------------------------------------------
    def get_sub_cycles_NN(self):

        ## regarde si le fichier existe dejà :
        fileName = self.to_save + self.signaltype + '_train_sub_cycles_NN_1' + '.npy'
        fileObj = Path(fileName)

        is_fileObj = fileObj.is_file()

        tosave_train_cycle_NN = self.to_save + self.signaltype + '_train_sub_cycles_NN'
        tosave_val_cycle_NN = self.to_save + self.signaltype + '_val_sub_cycles_NN'
        tosave_test_cycle_NN = self.to_save + self.signaltype + '_test_sub_cycles_NN'

        if is_fileObj:
            print('NN cycles déjà enregisté')

            recup_train = Cell(tosave_train_cycle_NN, self.config.nb_set)
            recup_val = Cell(tosave_val_cycle_NN, self.config.nb_set)
            recup_test = Cell(tosave_test_cycle_NN, self.config.nb_set)

            train_sub_cycle_NN = recup_train.reco_cell()
            val_sub_cycle_NN = recup_val.reco_cell()
            test_sub_cycle_NN = recup_test.reco_cell()

        else:
            train_sub_cycle_NN = [0 for i in range(self.config.nb_set)]
            val_sub_cycle_NN = [0 for i in range(self.config.nb_set)]
            test_sub_cycle_NN = [0 for i in range(self.config.nb_set)]

            for i in range(self.config.nb_set):
                mask_train = [self.config.sub_cycles[i][j] in self.train_cycle for j in range(self.config.nbcycle[i])]
                mask_val = [self.config.sub_cycles[i][j] in self.val_cycle for j in range(self.config.nbcycle[i])]
                mask_test = [self.config.sub_cycles[i][j] in self.test_cycle for j in range(self.config.nbcycle[i])]

                train_sub_cycle_NN[i] = np.asarray(self.config.sub_cycles[i])[mask_train]
                val_sub_cycle_NN[i] = np.asarray(self.config.sub_cycles[i])[mask_val]
                test_sub_cycle_NN[i] = np.asarray(self.config.sub_cycles[i])[mask_test]

        return train_sub_cycle_NN, val_sub_cycle_NN, test_sub_cycle_NN

    # ------------------------------------------
    def create_NN_data(self, f, ext, t, index_picture, number_picture, NN_cycles, numbertot_picture=None, sub_cycles=None):

        f_NN = f[NN_cycles, :]
        ext_NN = ext[NN_cycles, :]
        t_NN = t[NN_cycles, :]

        if self.config.img:
            index_picture_NN = index_picture[NN_cycles, :]
            number_picture_NN = number_picture[NN_cycles, :]
            if self.config.mix_set:
                numbertot_picture_NN = numbertot_picture[NN_cycles, :]
                nb_index_picture_NN = np.zeros(self.config.nb_set)
                for j in range(self.config.nb_set):
                    # print(sub_cycles[j])
                    for i in sub_cycles[j]:
                        # print(i)
                        nb_index_picture_NN[j] = nb_index_picture_NN[j] + np.sum(index_picture[i, :])
            else:
                numbertot_picture_NN = None
                nb_index_picture_NN = 0
                for j in range(np.size(NN_cycles)):
                    where = np.where(index_picture_NN[j, :] == 1)[0]
                    nb_index_picture_NN = nb_index_picture_NN + np.size(where)

        else:
            index_picture_NN = None
            number_picture_NN = None
            numbertot_picture_NN = None
            nb_index_picture_NN = 0

        return f_NN, t_NN, ext_NN, index_picture_NN, number_picture_NN, numbertot_picture_NN, nb_index_picture_NN