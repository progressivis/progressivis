# -*- coding: utf-8 -*-
""" Computes the distance matrix from each row of a data frame.
"""
from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances

import logging
logger = logging.getLogger(__name__)

class PairwiseDistances(DataFrameModule):
    def __init__(self, metric='euclidean', columns=None, n_jobs=1, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(PairwiseDistances, self).__init__(**kwds)
        self.default_step_size = kwds.get('step_Size', 100)  # initial guess        
        self._metric = metric
        self._n_jobs = n_jobs
        if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
            raise ProgressiveError('Unknown distance %s', metric)
        self.columns = columns

    def is_ready(self):
        if not (self.get_input_slot('df').is_buffer_empty()):
            return True
        return super(PairwiseDistances, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        import pdb
        pdb.set_trace()
        dfslot = self.get_input_slot('df')
        df = dfslot.data()
        dfslot.update(run_number, df)
        if len(dfslot.deleted) or len(dfslot.updated) > len(dfslot.created):
            dfslot.reset()
            logger.info('Reseting history because of changes in the input df')
            #TODO: be smarter with changed values
        dfslot.buffer_created()

        len_b = len(dfslot.created)
        n = len(df)-len_b

        if self.columns is None:
            self.columns = df.columns.delete(np.where(df.columns==DataFrameModule.UPDATE_COLUMN))
        elif not isinstance(self.columns, pd.Index):
            self.columns = pd.Index(self.columns)

        row = None
        try:
            rows = df[self.columns]
        except Exception as e:
            logger.error('While extracting columns', e)
            raise

        # We have the old matrix Si of size (n), we want to complete it with
        # two sub matrices, Sij and Sj of length (m).
        # We are given a "budget" of step_size (s) operations
        # See how many new rows we can compute with our budget.
        # These will be the (j) new rows
        # We need to comput Sij, which will take n*m operations
        # and Sj which will take m*m, so we search m for n*m + m*m = s
        # m^2 + n*m -s = 0, a=1, b=n, c=-s, solution is -b +- sqrt(b^2-4ac)/2a
        # The only positive solution is -n + sqrt(n^2+4s) / 2

        if n==0:
            m = int(np.sqrt(step_size))
        else:
            m = (-n + np.sqrt(n*n + 4*step_size)) / 2.0
            m = int(np.min([1.0, m]))

        Si = self._df
        indices = dfslot.next_buffered(m)
        j = rows.loc[indices]
        Sj = pairwise_distances(j, metric=self._metric, n_jobs=self._n_jobs)
        if Si is None:
            S = Sj
            index = Si.index
        else:
            i = df[Si.index]
            Sij = pairwise_distances(i,j)
            Sji = Sij.T
            S1 = np.hstack((Si, Sij))
            S2 = np.hstack((Sji, Sj))
            S = np.vstack((S1, S2))
            index = Si.index + df.index[indices]
        self._df = pd.DataFrame(S,index=index)

def cosine_distance(X, Y=None):
    """
    Compute cosine distance between samples in X and Y.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).

    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    distance matrix : array
        An array with shape (n_samples_X, n_samples_Y).

    See also
    --------
    sklearn.metrics.pairwise.cosine_similarity
    scipy.spatial.distance.cosine (dense matrices only)
    """
    # 1.0 - cosine_similarity(X, Y) without copy
    S = cosine_similarity(X, Y)
    S *= -1
    S += 1
    return S


def cosine_similarity(X, Y=None):
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).

    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    kernel matrix : array
        An array with shape (n_samples_X, n_samples_Y).
    """
    # to avoid recursive import

    if Y is None:
        Y = X

    X_normalized = normalize(X)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y)

    if isinstance(X_normalized, pd.Series) and isinstance(Y_normalized, pd.Series):
        return (X_normalized*Y_normalized).sum()
    K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=True)

    return K

def normalize(X):
    if (isinstance(X,pd.Series) and X.dtype.hasobject):
        ret=[]
        normalized=True
        for x in X:
            if hasattr(x, 'normalized') and x.normalized:
                ret.append(x)
            else:
                sum = np.sqrt((x*x).sum())
                if np.isclose(sum, 1):
                    norm = x
                else:
                    norm = x/sum
                    normalized = False
                norm.normalized = True
                ret.append(norm)
        if normalized:
            return X
        return pd.Series(ret, index=X.index, dtype=np.object)
    return sk_normalize(X)
        
