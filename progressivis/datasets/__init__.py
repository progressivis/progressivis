import os

from progressivis import ProgressiveError

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))

def get_dataset(name, **kwds):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    if name=='bigfile':
        return generate_random_csv('%s/bigfile.csv'%data_dir, 1000000, 30)
    if name=='smallfile':
        return generate_random_csv('%s/smallfile.csv'%data_dir, 30000, 10)
    if name=='warlogs':
        return wget_file(filename='%s/warlogs.vec.bz2'%data_dir,
                         url='http://www.cs.ubc.ca/labs/imager/video/2014/QSNE/warlogs.vec.bz2')
    if name.startswith('cluster:'):
        fname = name[len('cluster:'):] + ".txt"
        return wget_file(filename='%s/%s'%(data_dir,fname),
                         url='http://cs.joensuu.fi/sipu/datasets/%s'%fname)
    raise ProgressiveError('Unknow dataset %s', name)

from .random import generate_random_csv
from .wget import wget_file


__all__ = ['get_dataset',
           'generate_random_csv']

