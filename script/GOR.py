#! 

""" This is where the manual goes
    """

import os
import numpy as np
from Utility.Utilis import Pssm, Dssp, Stats
from math import log2
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import classification_report, accuracy_score
import joblib

class GOR():

    def __init__(self, tr_db_dir, tes_dir, cv = None, tes_id_path = None, tr_id_path = None, window=17, out = None):
        # init directories
        self.directory = tr_db_dir
        self.tr_id_path = tr_id_path
        self.directorytest = tes_dir
        self.te_id_path = tes_id_path
        self.out = out
        self.cv = cv

        # init model mx
        self.window = window
        self.pad = self.window // 2
        self.E = np.zeros([self.window, 20], dtype=float)
        self.H = np.zeros([self.window, 20], dtype=float)
        self.C = np.zeros([self.window, 20], dtype=float)
        self.r = np.zeros([self.window, 20], dtype=float)
        self.SS = {}

        # init results
        self.model = self.training().information_mx()
        self.y_pred =self.predict()
        self.dict_pred = {}

    @staticmethod
    def get_pssmids(dir):

        IDsPssm = []
        k = 0

        for filename in sorted(os.listdir(dir)):
            if filename.endswith('.pssm'):
                k += 1
                if Pssm.check_empty(Pssm(filename[:-5],dir).parse()):
                    IDsPssm.append(filename[:-5])

        return IDsPssm

    def training(self, id_l = None):

        if self.tr_id_path != None and id_l == None:
            with open(self.tr_id_path, 'r') as f:
                id_file = f.readlines()
                id_list = [x.strip() for x in id_file]

        elif self.tr_id_path == None and id_l == None:
            id_list = []

            for filename in sorted(os.listdir(self.directory)):
                if filename.endswith('.pssm'):
                    if filename[:-5] not in id_list:
                        id_list.append(filename[:-5])

        else:
            id_list_dir = self.get_pssmids(self.directory)
            id_list = list(set(id_l).intersection(id_list_dir))


        for id in id_list:

            # parsa il file pssm ed estrae il profilo
            prf = Pssm(id, self.directory).parse()

            # parsa il file dssp ed estrae la sequenza amminoacidica
            ss_seq = Dssp(id, self.directory).parse_dssp()

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

    def information_mx(self):

        self.norm()

        self.infE = np.zeros([self.window, 20], dtype=float)
        self.infH = np.zeros([self.window, 20], dtype=float)
        self.infC = np.zeros([self.window, 20], dtype=float)


        for i in range(self.window) :
            for k in range(20):
                self.infE[i][k] = log2(self.E[i][k] / (float(self.SS['E']) * self.r[i][k]))
                self.infC[i][k] = log2(self.C[i][k] / (float(self.SS['-']) * self.r[i][k]))
                self.infH[i][k] = log2(self.H[i][k] / (float(self.SS['H']) * self.r[i][k]))

        if self.out != None:
            self.Inf = np.concatenate((self.infH, self.infE, self.infC))
            np.savetxt(self.out + '/Inf_mx' + ".tsv", self.Inf, delimiter="\t")
        return self

    def predict(self, id_lis = None):

        dir = self.directorytest
        dict_out = {}
        l = '210'

        if self.te_id_path != None and id_lis == None:
            with open(ids_path_file, 'r') as f :
                id_file = f.readlines()
                id_l = [x.strip() for x in id_file]

        elif self.te_id_path == None and id_lis == None:
            id_l = []

            for filename in sorted(os.listdir(dir)) :
                if filename.endswith('.pssm') :
                    if filename[:-5] not in id_l:
                        id_l.append(filename[:-5])

        else:
            id_list_dir = self.get_pssmids(self.directory)
            id_l = list(set(id_lis).intersection(id_list_dir))

        sseq = ''
        for ID in id_l:
            seq_name = '>' + ID

            # parsa il file pssm ed estrae il profilo
            prf = Pssm(ID, dir).parse()

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
                sseq += seq_pred

        if self.out != None:
            i = 0
            with open(self.out + '/predicted_ss.fasta', 'w') as pred_out:
                for k in sorted(dict_out.keys()):
                    i += 1
                    pred_out.write(k + '\n' + dict_out[k]+'\n')
            print(i)

        self.y_pred = list(map(int,sseq))
        self.dict_pred = dict_out

        return self

    @staticmethod
    def getytrue(dir,name):

        seq = Dssp(name, dir).parse_dssp()
        o = seq.translate(str.maketrans({'-' : '2', 'E' : '1', 'H' : '0'}))
        y_true = list(list(map(int, o)))
        return y_true

    def clsreport(self):
        ss = ['E', 'C', 'H']
        y_pred = self.y_pred
        self.y_true = []
        for key in self.dict_pred:
            self.y_true += self.getytrue(self.directorytest, key[1:])

        mcc = Stats.MCC(self.y_true, y_pred, True)
        self.report = classification_report(self.y_true, y_pred,
                                       target_names= ['H', 'E', 'C'],
                                       output_dict=True)

        self.report['ACC'] = accuracy_score(self.y_true, y_pred)
        self.report['mclasscm'] = Stats.multiclasscm(self.y_true, y_pred)

        for key in mcc:
            self.report[key]['MCC'] = mcc[key]

        print('ACC: {} \nConfMx: {}'.format(self.report['ACC'], self.report['mclasscm']))
        for key in ss :
            print('{0}\tprecision\t{1:0.5f}\trecall\t{2:0.5f}\tMCC\t{3:0.5f}'.format(key,
                                                                                     self.report[key]['precision'],
                                                                                     self.report[key]['recall'],
                                                                                     self.report[key]['MCC']))
        return self.report

    def crossval(self):
        if self.cv != None:
            self.directorytest = directory_train
            cv = []
            ts = []
            c = 0
            for file in sorted(os.listdir(dir)) :
                if file.__contains__('set'):
                    with open(dir+file, 'r') as f:
                        lines = f.readlines()
                        Set = [x.strip() for x in lines]
                        ts += [c for i in range(len(Set))]
                        cv += Set
                        c += 1

            cv = np.array(cv)
            ps = PredefinedSplit(test_fold=ts)
            fold = 1
            for train_index, test_index in ps.split():
                self.model = self.training(list(cv[train_index])).information_mx()
                self.pred = self.predict(list(cv[test_index]))

                print('FOLD{}:'.format(fold))
                self.report = self.model.clsreport()
                fold += 1

            return self


if __name__ == '__main__':

    import timeit
    import os

    start = timeit.default_timer()

    directory_train = '../training_file'
    directory_test = '../blindset/test_file'

    dir = '../cv/'

    model = GOR(directory_train, directory_test, cv = dir, window=17)

    model.crossval()
    model.predict().clsreport()

    print('Wait a sec! I\' gonna save your model in a safe place!')
    joblib.dump(model.__dict__, 'GORmodel.joblib')

    stop = timeit.default_timer()

    print('Time: ', stop - start)

