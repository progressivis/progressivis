import os

from progressivis import ProgressiveError

def get_dataset(name):
    if not os.path.isdir('data'):
        mkdir('data')
    if name=='bigfile':
        return generate_random_csv('data/bigfile.csv', 1000000, 30)
    if name=='smallfile':
        return generate_random_csv('data/smallfile.csv', 30000, 10)
    if name=='warlogs':
        return wget_file(filename='data/warlogs.vec.bz2',
                         url='http://www.cs.ubc.ca/labs/imager/video/2014/QSNE/warlogs.vec.bz2')
    raise ProgressiveError('Unknow dataset %s', name)

from .random import generate_random_csv
from .wget import wget_file


__all__ = ['get_dataset',
           'generate_random_csv']

