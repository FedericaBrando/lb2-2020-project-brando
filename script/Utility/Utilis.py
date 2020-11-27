#! 

""" This is where the manual goes
    """
import numpy as np


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
            np_prof = np.vstack(mx)

        return np_prof



class Dssp():

    def __init__(self, filename, directory):
        self.filename = filename + '.dssp'
        self.directory = directory
        self.path = self.directory + '/' + self.filename

    def parse_dssp(self):
        with open(self.path) as fdssp:
            fdssp.readline()
            return fdssp.readline().rstrip()