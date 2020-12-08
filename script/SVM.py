#! 

""" This is where the manual goes
    """
import os
from sklearn.model_selection import PredefinedSplit,cross_validate
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from Utility.Utilis import Pssm, Dssp, Stats
import numpy as np
import joblib

class SVM_ss:

    def __init__(self, window, pssm_directory, out_dir, cv_dir = 'None'):
        self.w = window
        self.pad = window // 2
        self.cv_dir = cv_dir
        self.dir = pssm_directory
        self.o = out_dir
        self.model = self.predict()

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

    def one_input(self, dir, id):

        # file DSSP
        seq = Dssp(id, dir).parse_dssp()
        o = seq.translate(str.maketrans({'-' : '2', 'E' : '1', 'H' : '0'}))
        ss = list(list(map(int, o)))

        # file PSSM
        prf = Pssm(id, dir).parse()
        tmp = np.zeros([self.pad, 20], dtype=float)
        prf = np.concatenate((tmp, prf, tmp))

        mx = []

        p = 0
        r = self.w
        while r <= len(prf) :
            x = prf[p:r].flatten().tolist()
            mx.append(x)
            p += 1
            r += 1

        return mx, ss

    def datasets_parse(self, ids_file ='None'):
        self.ids_file = ids_file


        if ids_file != 'None':
            with open(ids_file, 'r') as f:
                id_file = f.readlines()
                id_l = [x.strip() for x in id_file]

            id_list_folder = SVM_ss.get_pssmids(self.dir)
            id_list = list(set(id_l).intersection(id_list_folder))
        else:
            id_list = SVM_ss.get_pssmids(self.dir)

        X = []
        y = []

        j=0
        count=0

        while j != len(id_list):
            X_tmp, y_tmp = self.one_input(self.dir, id_list[j])
            X += X_tmp
            y += y_tmp
            count += 1

            j+=1

        print('ID utilizzabili: ', count)

        return X, y

    def cv_extract(self):
        X = []
        y = []
        test_set = []
        c = 1
        for file in list(sorted(os.listdir(self.cv_dir))):
            if file.__contains__('set'):
                X_tmp, y_tmp = self.datasets_parse(self.cv_dir+file)
                mapping = [c for i in range(len(X_tmp))]
                c += 1
                test_set += mapping
                X += X_tmp
                y += y_tmp
                # print(len(X_tmp), len(y_tmp))

        print(len(test_set), len(X), len(y))
        return X,y,test_set

    def predict(self):
        if self.cv_dir != 'None':
            X, y, ts = self.cv_extract()
            X = np.array(X)
            y = np.array(y)
            mcc_mean = make_scorer(Stats.MCC)
            split = PredefinedSplit(ts)
            model = SVC(kernel='rbf', gamma=2, C=2, random_state=42, verbose=10)

            scores = cross_validate(model, X, y,
                                    cv=split,
                                    return_estimator=True,
                                    scoring=mcc_mean,
                                    verbose=10,
                                    n_jobs= -1)
            print(scores)
            return scores

    def dump(self, name):
        filename = os.path.join(self.o, name+'.joblib')
        joblib.dump(self.model, filename)


if __name__ == '__main__':

    import timeit

    start = timeit.default_timer()
    dir_cv = '../test/test2/'

    SVM_ss(17, '../training_file',dir_cv ,dir_cv).predict()

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # print(pr)

    # print(pr,seq, sep='\n')

