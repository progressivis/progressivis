from progressivis import DataFrameModule, SlotDescriptor

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

import logging
logger = logging.getLogger(__name__)

class MBKMeans(DataFrameModule):
    """
    Mini-batch k-means using the sklearn implementation.
    """
    schema = [('array', np.dtype(object), None),
              DataFrameModule.UPDATE_COLUMN_DESC]

    def __init__(self, columns, n_clusters, batch_size, **kwds):
        self.mbk = MiniBatchKMeans()
        self._add_slots(kwds, 'input_descriptors',
                [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(MBKMeans, self).__init__(dataframe_slot='df', **kwds)
        self.columns = columns
        self.n_clusters = n_clusters
        self._df = self.create_dataframe(MBKMeans.schema)

    def is_ready(self):
        return super(MBKMeans, self).is_ready()

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)

        if dfslot.has_updated() or dfslot.has_deleted():
            #No need to act (mbkmeans does not need to keep track of changes)
            pass

        input_df = dfslot.data()
        if step_size < len(dfslot.data()):
            self.mbk.fit(X=input_df.as_matrix(columns=self.columns))
        else:
            self.mbk.fit_partial(X=input_df.as_matrix(columns=self.columns))

        self._df = self.get_cluster_centers_df()
        return self._return_run_step(dfslot.next_state(), steps_run=len(dfslot.data))

    def get_cluster_centers(self):
        try:
            return self.mbk.cluster_centers_ 
        except AttributeError:
            return pd.Series([])

    def get_cluster_centers_df(self):
        try:
            return pd.DataFrame(self.mbk.cluster_centers_, columns=self.columns)
        except AttributeError:
            return pd.DataFrame([], columns=self.columns)

    def is_visualization(self):
        return False

    def to_json(self, short=False):
        json = super(MBKMeans, self).to_json(short)
        #if short:
        #    return json
        return self._centers_to_json(json)

    def _centers_to_json(self, json):
        json['cluster_centers'] = self.get_cluster_centers_df().to_json()
        return json

