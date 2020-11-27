# Python script by fecke 25/10/20 21:11

def normalize(list):
    newlist = []
    # print(list)
    for l in list[:]:
        newlist.append(str(int(l) / 100))
    return newlist


def parse_pssm(filename):
    with open(filename) as f:
        f = f.readlines()
        aa = f[2].split()
        print('\t'.join(aa[-20:]))
        with open(filename[:-5] + '.tsv', 'w') as o:
            o.write('Num\t' + '\t'.join(aa[-20:]) + '\n')
            for line in f[3:-6]:
                l = line.split()
                o.write('\t'.join(l[0 :1]) + '\t' + '\t'.join(normalize(l[22 :-2])) + '\n')


if __name__ == '__main__':
    import os
    import sys

    # parse_pssm('./psiblast_training/e1n13.1A.pssm')
    
    try:
        dir_path = sys.argv[1]
        for filename in sorted(os.listdir(dir_path)):
            if filename.endswith('.pssm'):
                fn = dir_path + filename
                parse_pssm(fn)
    except IndexError:
        print('\nMissing argument: directory path')
