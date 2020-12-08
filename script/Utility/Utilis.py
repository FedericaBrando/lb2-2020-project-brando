#! 

""" This is where the manual goes
    """
import numpy as np
from statistics import mean
from sklearn.metrics import matthews_corrcoef, accuracy_score, classification_report

import math

class Pssm:

    def __init__(self, filename, directory):
        self.filename = filename + '.pssm'
        self.directory = directory
        self.path = self.directory+"/"+self.filename

    @staticmethod
    def normalize(lista):
        newlist = []
        for l in lista[:]:
            newlist.append(float(l) / 100)
        return newlist

    @staticmethod
    def check_empty(np_array):
        if np.count_nonzero(np_array) == 0:
            return False
        return True

    def parse(self):
        with open(self.path) as pssm:
            pssm = pssm.readlines()
            mx = []
            
            for line in pssm[3 :-6] :
                l = line.split()
                mx.append(Pssm.normalize(l[22:-2]))
            self.np_prof = np.vstack(mx)

        return self.np_prof

    def svm_parse(self):

        prf_list = []


        for row in self.np_prof:
            for col in self.np_prof:
                prf_list.append(self.np_prof[row][col])




class Dssp():

    def __init__(self, filename, directory):
        self.filename = filename + '.dssp'
        self.directory = directory
        self.path = self.directory + '/' + self.filename

    def parse_dssp(self):
        with open(self.path) as fdssp:
            fdssp.readline()
            return fdssp.readline().rstrip()


class Stats():

    def __init__(self):
        pass

    @staticmethod
    def MCC(y_true, y_pred, single = False):
        sslist = [0, 1, 2]
        ss = 'HEC'

        mcc = {}
        for ssclass in sslist :
            yp_tmp = [ss if ss == ssclass else 9 for ss in y_pred]
            yt_tmp = [ss if ss == ssclass else 9 for ss in y_true]
            mcc_tmp = matthews_corrcoef(yp_tmp, yt_tmp)
            mcc[ss[ssclass]] = mcc_tmp

        meanMCC = mean(mcc.values())

        if single:
            return mcc

        else: return meanMCC

    @staticmethod
    def ACC(y_true, y_pred, single = False):
        sslist = [0, 1, 2]

        acc = []
        for ssclass in sslist :
            yp_tmp = [ss if ss == ssclass else 9 for ss in y_pred]
            yt_tmp = [ss if ss == ssclass else 9 for ss in y_true]
            acc_tmp = accuracy_score(yp_tmp, yt_tmp)
            acc.append(acc_tmp)
        meanACC = mean(acc)

        if single :
            return acc

        else :
            return meanACC

if __name__ == '__main__':

    yt = np.array([0,0,2,1,2,0,1,0,1,1,1,1])
    yp = np.array([0,0,0,1,2,0,1,0,1,1,1,1])

    Stats.clsreport(yt,yp)