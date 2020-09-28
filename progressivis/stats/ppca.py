import numpy as np
import copy
from ..core.utils import indices_len, fix_loc, filter_cols
from ..core.bitmap import bitmap
from ..table.module import TableModule
from ..table import Table, TableSelectedView
from ..table.dshape import dshape_projection
from ..core.decorators import *
from .. import ProgressiveError, SlotDescriptor
from ..utils.psdict import PsDict
from . import Sample
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from scipy.spatial import distance as dist
import numexpr as ne

class PPCA(TableModule):
    parameters = [('n_components',  np.dtype(int), 2)]    
    inputs = [SlotDescriptor('table', type=Table, required=True)]
    outputs = [SlotDescriptor('transformer', type=PsDict, required=False)]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.inc_pca = None # IncrementalPCA(n_components=self.params.n_components)
        self.inc_pca_wtn = None
        self._transformer = PsDict()
        self.default_step_size = 10000
        
    def reset(self):
        print("RESET PPCA")
        self.inc_pca = IncrementalPCA(n_components=self.params.n_components)
        self.inc_pca_wtn = None
        if self._table is not None:
            self._table.selection = bitmap()

    def get_data(self, name):
        if name == 'transformer':
            return self._transformer
        return super().get_data(name)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        """
        """    
        with self.context as ctx:
            #import pdb;pdb.set_trace()
            table = ctx.table.data()
            indices = ctx.table.created.next(step_size) # returns a slice
            steps = indices_len(indices)
            if steps < self.params.n_components:
                return self._return_run_step(self.state_blocked, steps_run=0)

            vs = self.filter_columns(table, fix_loc(indices))
            vs = vs.to_array()
            if self.inc_pca is None:
                self.inc_pca = IncrementalPCA(n_components=self.params.n_components)
                self._transformer['inc_pca'] = self.inc_pca
            self.inc_pca.partial_fit(vs)
            if self._table is None:
                self._table = TableSelectedView(table, bitmap(indices))
            else:
                self._table.selection |= bitmap(indices)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)

    def create_dependent_modules(self, atol=0.0, rtol=0.001, trace=False):
        scheduler = self.scheduler()
        with scheduler:
            self.reduced = PPCATransformer(scheduler=scheduler,
                                           atol=atol, rtol=rtol, trace=trace, group=self.name)
            self.reduced.input.table = self.output.table
            #import pdb;pdb.set_trace()
            self.reduced.input.transformer = self.output.transformer
            self.reduced.create_dependent_modules(self.output.table)

class PPCATransformer(TableModule):
    inputs = [SlotDescriptor('table', type=Table, required=True),
              SlotDescriptor('samples', type=Table, required=True),
              SlotDescriptor('transformer', type=PsDict, required=True)]

    def __init__(self, atol=0.0, rtol=0.001, trace=False, **kwds):
        super().__init__(**kwds)
        self._atol = atol
        self._rtol = rtol
        self._trace = trace
        self._trace_df = None
        self.inc_pca_wtn = None
        self._table = None

    def create_dependent_modules(self, input_slot):
        scheduler = self.scheduler()
        with scheduler:
            self.sample = Sample(samples=100, group=self.name,
                                     scheduler=scheduler)
            self.sample.input.table = input_slot
            self.input.samples = self.sample.output.select

    def trace_if(self, ret, mean, max_, len_):
        if self._trace:
            row = dict(Action="RESET" if ret else "PASS",
                       Mean=mean, Max=max_, Length=len_)
            if self._trace_df is None:
                self._trace_df = pd.DataFrame(row, index=[0])
            else:
                self._trace_df = self._trace_df.append(row, ignore_index=True)
            if self._trace == "verbose":
                print(row)
            """if ret:
                print(f"RESET, {mean:.4f}<={self._rtol}, {max_:.4f} data length :{len_}")
            else:
                print(f"FINE, {mean:.4f}<={self._rtol}, {max_:.4f} data length :{len_}")
            """
        return ret

    def needs_reset(self, inc_pca, inc_pca_wtn, input_table, samples):
        data = self.filter_columns(input_table, samples).to_array()
        transf_wtn = inc_pca_wtn.transform(data)
        transf_now = inc_pca.transform(data)
        explained_variance = inc_pca.explained_variance_
        dist = np.sqrt(ne.evaluate("((transf_wtn-transf_now)**2)/explained_variance").sum(axis=1))
        mean = np.mean(dist)
        max_ = np.max(dist)
        ret = mean > self._rtol
        return self.trace_if(ret, mean, max_, len(input_table))
            
    def reset(self):
        if self._table is not None:
            self._table.resize(0)

    @process_slot("table", reset_cb="reset")
    @process_slot("samples", reset_if=False)
    @process_slot("transformer", reset_if=False)    
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        """
        """    
        with self.context as ctx:
            input_table = ctx.table.data()
            indices = ctx.table.created.next(step_size) # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            transformer = ctx.transformer.data()
            ctx.transformer.clear_buffers()
            inc_pca = transformer.get('inc_pca')
            ctx.samples.clear_buffers()
            if self.inc_pca_wtn is not None:
                samples = ctx.samples.data()
                if self.needs_reset(inc_pca, self.inc_pca_wtn, input_table, samples):
                    self.inc_pca_wtn = None
                    ctx.table.reset()
                    ctx.table.update(run_number)
                    self.reset()
                    indices = ctx.table.created.next(step_size)
                    steps = indices_len(indices)
                    if steps == 0:
                        return self._return_run_step(self.state_blocked, steps_run=0)
            else:
                self.inc_pca_wtn = copy.deepcopy(inc_pca)
            data = self.filter_columns(input_table, fix_loc(indices)).to_array()
            reduced = inc_pca.transform(data)
            if self._table is None:
                cols = [f"_pc{i}" for i in range(reduced.shape[1])]
                df = pd.DataFrame(reduced, columns=cols)
                self._table = Table(self.generate_table_name('ppca'),
                                    data=df, create=True)
            else:
                self._table.append(reduced)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)
