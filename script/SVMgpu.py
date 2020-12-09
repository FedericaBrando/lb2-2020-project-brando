#! 

""" This is where the manual goes
    """
import os
from sklearn.model_selection import PredefinedSplit,cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, classification_report, accuracy_score, multilabel_confusion_matrix
from Utility.Utilis import Pssm, Dssp, Stats
import numpy as np
import joblib
from thundersvm import SVC

class SVM_ss:

    def __init__(self, window, pssm_directory,
                 out_dir = './',
                 cv_dir = 'None'):

        self.w = window
        self.pad = window // 2
        self.cv_dir = cv_dir
        self.traindir = pssm_directory
        self.o = out_dir

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

    def datasets_parse(self, pssmdir, ids_file ='None'):
        self.ids_file = ids_file


        if ids_file != 'None':
            with open(ids_file, 'r') as f:
                id_file = f.readlines()
                id_l = [x.strip() for x in id_file]

            id_list_folder = SVM_ss.get_pssmids(pssmdir)
            id_list = list(set(id_l).intersection(id_list_folder))

        else:
            id_list = SVM_ss.get_pssmids(pssmdir)

        X = []
        y = []

        j=0
        count=0

        while j != len(id_list):
            X_tmp, y_tmp = self.one_input(pssmdir, id_list[j])
            X += X_tmp
            y += y_tmp
            count += 1

            j+=1

        print('ID utilizzabili: ', count)
        print(len(X))

        return X, y

    def cv_extract(self):
        X = []
        y = []
        test_set = []
        c = 1
        for file in list(sorted(os.listdir(self.cv_dir))):
            if file.__contains__('set'):
                X_tmp, y_tmp = self.datasets_parse(self.traindir, self.cv_dir+file)
                mapping = [c for i in range(len(X_tmp))]
                c += 1
                test_set += mapping
                X += X_tmp
                y += y_tmp
                # print(len(X_tmp), len(y_tmp))

        print(len(test_set), len(X), len(y))
        return X,y,test_set

    def gridsearchCV(self):
        print(self.cv_dir)
        if self.cv_dir != 'None':
            X, y, ts = self.cv_extract()
            X = np.array(X)
            y = np.array(y)
            mcc_mean = make_scorer(Stats.MCC)
            split = PredefinedSplit(ts)
            model = SVC(kernel='rbf', gamma=2, C=2,
                        random_state=42,
                        verbose=True, gpu_id=0,
                        cache_size=5000)

            hyper_params = [ {'C': [2,4],
                     'gamma': [0.5,2]}]

            self.model_cv = GridSearchCV(estimator=model,
                                    param_grid=hyper_params,
                                    scoring=mcc_mean,
                                    cv=split,
                                    verbose=10,
                                    return_train_score=True,
                                    n_jobs=1).fit(X,y)
            print(self.model_cv.cv_results_)

            joblib.dump(self.model_cv.__dict__, 'gridsearch.joblib')

            print('best params: ', self.model_cv.best_params_)
            bp = list(self.model_cv.best_params_.values())
            self.C, self.gamma = bp
            return self

        else:

            self.gamma = 0.5
            self.C = 2
            print('No directory for cross validation:\nset parameters to gamma:{} and C:{}'.format(self.gamma, self.C))
            return self


    # def dump(self, name):
    #     filename = os.path.join(self.o, name+'.joblib')
    #     joblib.dump(self.model, filename)

    def predict(self, test_dir):
        X_train, y_train = self.datasets_parse(self.traindir)
        X_test, y_test = self.datasets_parse(test_dir)

        self.gridsearchCV()

        s = timeit.default_timer()
        model = SVC(kernel='rbf', gamma=self.gamma, C=self.C,
                    random_state=42,
                    verbose=True, gpu_id=0,
                    cache_size=5000)

        model.fit(X_train,y_train)

        e = timeit.default_timer()

        print('Fine training: ', e - s)
        y_pred = model.predict(X_test)
        self.report = classification_report(y_test, y_pred,
                                       target_names= ['H', 'E', 'C'],
                                       output_dict=True)

        self.report['ACC'] = accuracy_score(y_test, y_pred)
        mcc = Stats.MCC(y_test,y_pred, True)
        acc = Stats.ACC(y_test,y_pred, True)

        for key in mcc :
            self.report[key]['MCC'] = mcc[key]
            self.report[key]['ACC'] = acc[key]


        ss = ['H', 'E', 'C']
        print('ACC: {}'.format(self.report['ACC']))
        for key in ss :
            print('{0}\tprecision\t{1:0.5f}\trecall\t{2:0.5f}\tMCC\t{3:0.5f}'.format(key,
                                                                                     self.report[key]['precision'],
                                                                                    self.report[key]['recall'],
                                                                                     self.report[key]['MCC']))
        self.mx = multilabel_confusion_matrix(y_test,y_pred)
        self.multmx = Stats.multiclasscm(y_test, y_pred)
        print(self.mx, self.multmx, sep='\n')


        return self


if __name__ == '__main__':

    import timeit

    start = timeit.default_timer()
    dir_cv = '../cv/'
    train_pssm_dir = '../training_file'
    test_pssm_dir = '../blindset/test_file'

    model = SVM_ss(17, train_pssm_dir, cv_dir=dir_cv)

    model.predict(test_pssm_dir)

    print('I\'m storing the model in a safe place! Gimme a little more time')

    filename = 'SVM_model.joblib'
    joblib.dump(model.__dict__, filename)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
