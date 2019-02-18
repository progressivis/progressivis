import numpy as np
from ..table.module import TableModule
from ..table import Table
from ..core.utils import indices_len
from progressivis import ProgressiveError, SlotDescriptor
try:
    import pynene
    from .knnkde import KNNKernelDensity
except:
    pass

class KernelDensity(TableModule):
    parameters = [('samples',  object, 1), ('bins',  np.dtype(int), 1),]

    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                            [SlotDescriptor('table', type=Table, required=True),
                             ])
        self._kde = None
        self._json_cache = {}
        self._inserted = 0
        super(KernelDensity, self).__init__(**kwds)
    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('table')
        dfslot.update(run_number)
        if dfslot.deleted.any():
            #self.reset()
            #dfslot.update(run_number)
            raise ValueError("Not implemented yet")
        if not dfslot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        indices = dfslot.created.next(step_size, as_slice=False)
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self._kde is None:
            self._kde = KNNKernelDensity(dfslot.data(), online=True)
        res = self._kde.run(steps)
        self._inserted = res['numPointsInserted']
        samples = self.params.samples
        sampleN = self.params.bins
        scores = self._kde.score_samples(samples.astype(np.float32), k=100)
        self._json_cache = {
            'points': np.array(dfslot.data().loc[:500, :].to_dict(orient='split')['data']).tolist(),
            'bins': sampleN,
            'inserted': self._inserted,
            'total': len(dfslot.data()),
            'samples': [
                (sample, score) for sample, score in zip(samples.tolist(), scores.tolist())
            ]
        }
        return self._return_run_step(self.state_ready, steps_run=steps)
    def is_visualization(self):
        return True

    def get_visualization(self):
        return "knnkde"

    def to_json(self, short=False):
        json = super(KernelDensity, self).to_json(short)
        if short:
            return json
        return self.knnkde_to_json(json)
    def knnkde_to_json(self, json):
        json.update(self._json_cache)
        return json
