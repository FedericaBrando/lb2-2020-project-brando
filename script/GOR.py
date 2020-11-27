#! 

""" This is where the manual goes
    """

import os
import numpy as np
from Utility.Utilis import Pssm, Dssp
from math import log

class GOR():

    def __init__(self, window=17):
        self.window = window
        self.pad = self.window // 2
        self.E = np.zeros([self.window, 20], dtype=float)
        self.H = np.zeros([self.window, 20], dtype=float)
        self.C = np.zeros([self.window, 20], dtype=float)
        self.r = np.zeros([self.window, 20], dtype=float)
        self.SS = {}

    def training(self, database_directory):

        self.directory = database_directory

        id_list = []

        for filename in sorted(os.listdir(self.directory)):
            if filename.endswith('.pssm'):
                if filename[:-5] not in id_list:
                    id_list.append(filename[:-5])

                    # parsa il file pssm ed estrae il profilo
                    prf = Pssm(filename[:-5], self.directory).parse()

                    # parsa il file dssp ed estrae la sequenza amminoacidica
                    ss_seq = Dssp(filename[:-5], self.directory).parse_dssp()

                    if Pssm.check_empty(prf):

                        tmp = np.zeros([self.pad, 20], dtype=float)
                        prof = np.concatenate((tmp, prf, tmp))

                        ss = 0
                        p = 0
                        q = self.window

                        while q != len(prof)+1 and ss != len(ss_seq)+1 :

                            if ss_seq[ss] == '-' :
                                self.C += prof[p :q]
                                self.r += prof[p :q]
                                self.SS[ss_seq[ss]] = self.SS.get(ss_seq[ss], 0) + 1
                                ss += 1
                                p += 1
                                q += 1
                            elif ss_seq[ss] == 'H' :
                                self.H += prof[p :q]
                                self.r += prof[p :q]
                                self.SS[ss_seq[ss]] = self.SS.get(ss_seq[ss], 0) + 1
                                ss += 1
                                p += 1
                                q += 1
                            elif ss_seq[ss] == 'E' :
                                self.E += prof[p :q]
                                self.r += prof[p :q]
                                self.SS[ss_seq[ss]] = self.SS.get(ss_seq[ss], 0) + 1
                                ss += 1
                                p += 1
                                q += 1

        return self


    def norm(self):

        for row in range(len(self.r)) :
            n = self.r[row].sum()
            self.E[row] = self.E[row] / n
            self.H[row] = self.H[row] / n
            self.C[row] = self.C[row] / n
            self.r[row] = self.r[row] / n

        tot = sum(self.SS.values())

        for key in self.SS :
            self.SS[key] /= tot


        return self

    def information_mx(self, out=False):
        self.norm()

        self.infE = np.zeros([self.window, 20], dtype=float)
        self.infH = np.zeros([self.window, 20], dtype=float)
        self.infC = np.zeros([self.window, 20], dtype=float)

        for i in range(self.window) :
            for k in range(20):
                self.infE[i][k] = log(self.E[i][k] / (self.SS['E'] * self.r[i][k]), 2)
                self.infC[i][k] = log(self.C[i][k] / (self.SS['-'] * self.r[i][k]), 2)
                self.infH[i][k] = log(self.H[i][k] / (self.SS['H'] * self.r[i][k]), 2)

        if out:
            self.Inf = np.concatenate((self.infH, self.infE, self.infC))
            np.savetxt(self.directory + '/Inf_mx' + ".tsv", self.Inf, delimiter="\t")
        return self

    def predict(self, database_dir, out=False):

        dir = database_dir
        dict_out = {}
        l = '-EH'
        id_list = []

        for filename in sorted(os.listdir(dir)):
            if filename.endswith('.pssm'):
                if filename[:-5] not in id_list:
                    id_list.append(filename[:-5])
                    seq_name = '>' + filename[:-5]

                    # parsa il file pssm ed estrae il profilo
                    prf = Pssm(filename[:-5], dir).parse()

                    if Pssm.check_empty(prf):

                        tmp = np.zeros([self.pad, 20], dtype=float)
                        prof = np.concatenate((tmp, prf, tmp))

                        p = 0
                        r = self.window
                        seq_pred = ''

                        while r != len(prof) + 1:

                            p_C = np.sum(prof[p:r]*self.infC)
                            p_H = np.sum(prof[p:r]*self.infH)
                            p_E = np.sum(prof[p:r]*self.infE)
                            sspred = [p_C, p_E, p_H]
                            seq_pred = seq_pred + l[sspred.index(max(sspred))]

                            p, r = p + 1, r + 1

                        dict_out[seq_name] = seq_pred


        if out:
            i = 0
            with open(dir + '/predicted_ss.fasta', 'w') as pred_out:
                for k in sorted(dict_out.keys()):
                    i += 1
                    pred_out.write(k + '\n' + dict_out[k]+'\n')
            print(i)


        return dict_out


if __name__ == '__main__':
    import timeit
    import argparse

    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Predicting SS with GOR method')

    parser.add_argument('--dir_db_train', '-dirTr', type=str,
                        help='directory with .pssm and .dssp file of training set')
    parser.add_argument('--dir_db_test', '-dirTe', type=str,
                        help='directory with .pssm and .dssp file of test set')
    parser.add_argument('--window', '-win', type=int, help='size of the windows (int)')

    args = parser.parse_args()

    dirTr, w, dirTe = args.dir_db_train, args.window, args.dir_db_test

    model = GOR(window=w).training(dirTr).information_mx(out=True)

    model.predict(dirTe, True)

    # directory = '../training_file'
    # model = GOR(window=17).training(directory).information_mx(out=True)
    # output = model.predict('../blindset/psiblast_output', True)

    stop = timeit.default_timer()

    print('Time: ', stop - start)
