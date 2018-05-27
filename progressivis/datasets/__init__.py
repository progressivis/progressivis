import os

from progressivis import ProgressiveError
from .random import generate_random_csv
from .wget import wget_file

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))

def get_dataset(name, **kwds):
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    if name == 'bigfile':
        return generate_random_csv('%s/bigfile.csv'%DATA_DIR, 1000000, 30)
    if name == 'smallfile':
        return generate_random_csv('%s/smallfile.csv'%DATA_DIR, 30000, 10)
    if name == 'warlogs':
        return wget_file(filename='%s/warlogs.vec.bz2'%DATA_DIR,
                         url='http://www.cs.ubc.ca/labs/imager/video/2014/QSNE/warlogs.vec.bz2',
                         **kwds)
    if name.startswith('cluster:'):
        fname = name[len('cluster:'):] + ".txt"
        return wget_file(filename='%s/%s'%(DATA_DIR, fname),
                         url='http://cs.joensuu.fi/sipu/datasets/%s'%fname)
    raise ProgressiveError('Unknow dataset %s'%name)



__all__ = ['get_dataset',
           'generate_random_csv']
