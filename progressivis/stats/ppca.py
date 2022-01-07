from __future__ import annotations

import logging

import numpy as np
import copy

from ..core.utils import indices_len, fix_loc
from ..core.bitmap import bitmap
from ..table.module import TableModule, ReturnRunStep
from ..table import BaseTable, Table, TableSelectedView
from ..core.decorators import process_slot, run_if_any
from .. import SlotDescriptor
from ..utils.psdict import PsDict
from . import Sample
import pandas as pd
from sklearn.decomposition import IncrementalPCA  # type: ignore
import numexpr as ne  # type: ignore

from typing import Optional, Any, Dict, Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.slot import Slot


logger = logging.getLogger(__name__)


class PPCA(TableModule):
    parameters = [("n_components", np.dtype(int), 2)]
    inputs = [SlotDescriptor("table", type=Table, required=True)]
    outputs = [SlotDescriptor("transformer", type=PsDict, required=False)]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        # IncrementalPCA(n_components=self.params.n_components)
        self.inc_pca: Optional[IncrementalPCA] = None
        self.inc_pca_wtn: Optional[IncrementalPCA] = None
        self._as_array: Optional[str] = None
        self._transformer = PsDict()

    def predict_step_size(self, duration: float) -> int:
        p = super().predict_step_size(duration)
        return max(p, self.params.n_components + 1)

    def reset(self) -> None:
        logger.info("RESET PPCA")
        self.inc_pca = IncrementalPCA(n_components=self.params.n_components)
        self.inc_pca_wtn = None
        if self.result is not None:
            table = self.result
            assert isinstance(table, TableSelectedView)
            table.selection = bitmap()

    def get_data(self, name: str) -> Any:
        if name == "transformer":
            return self._transformer
        return super().get_data(name)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        """
        """
        assert self.context
        with self.context as ctx:
            table = ctx.table.data()
            indices = ctx.table.created.next(length=step_size)
            steps = indices_len(indices)
            if steps < self.params.n_components:
                return self._return_run_step(self.state_blocked, steps_run=0)

            vs = self.filter_columns(table, fix_loc(indices))
            if self._as_array is None:
                if len(vs.columns) == 1:
                    self._as_array = vs.columns[0]
                else:
                    self._as_array = ""
            avs = vs[self._as_array].values if self._as_array else vs.to_array()
            if self.inc_pca is None:
                self.inc_pca = IncrementalPCA(n_components=self.params.n_components)
                self._transformer["inc_pca"] = self.inc_pca
            self.inc_pca.partial_fit(avs)
            if self.result is None:
                self.result = TableSelectedView(table, bitmap(indices))
            else:
                table = self.result
                assert isinstance(table, TableSelectedView)
                table.selection |= bitmap(indices)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)

    def create_dependent_modules_buggy(
        self,
        atol=0.0,
        rtol=0.001,
        trace=False,
        threshold=None,
        resetter=None,
        resetter_slot="result",
    ):
        scheduler = self.scheduler()
        with scheduler:
            self.reduced = PPCATransformer(
                scheduler=scheduler,
                atol=atol,
                rtol=rtol,
                trace=trace,
                threshold=threshold,
                group=self.name,
            )
            self.reduced.input.table = self.output.result
            self.reduced.input.transformer = self.output.transformer
            if resetter is not None:
                resetter = resetter(scheduler=scheduler)
                resetter.input.table = self.output.result
                self.reduced.input.resetter = resetter.output[resetter_slot]
            self.reduced.create_dependent_modules(self.output.result)

    def create_dependent_modules(
        self,
        atol: float = 0.0,
        rtol: float = 0.001,
        trace=False,
        threshold: Optional[Any] = None,
        resetter: Optional[Any] = None,
        resetter_slot="result",
        resetter_func: Optional[Callable] = None,
    ):
        with self.tagged(self.TAG_DEPENDENT):
            s = self.scheduler()
            self.reduced = PPCATransformer(
                scheduler=s,
                atol=atol,
                rtol=rtol,
                trace=trace,
                threshold=threshold,
                resetter_func=resetter_func,
                group=self.name,
            )
            self.reduced.input.table = self.output.result
            self.reduced.input.transformer = self.output.transformer
            if resetter is not None:
                assert callable(resetter_func)
                self.reduced.input.resetter = resetter.output[resetter_slot]
            self.reduced.create_dependent_modules(self.output.result)


class PPCATransformer(TableModule):
    inputs = [
        SlotDescriptor("table", type=Table, required=True),
        SlotDescriptor("samples", type=Table, required=True),
        SlotDescriptor("transformer", type=PsDict, required=True),
        SlotDescriptor("resetter", type=PsDict, required=False),
    ]
    outputs = [
        SlotDescriptor("samples", type=Table, required=False),
        SlotDescriptor("prev_samples", type=Table, required=False),
    ]

    def __init__(
        self,
        atol: float = 0.0,
        rtol: float = 0.001,
        trace=False,
        threshold: Optional[Any] = None,
        resetter_func: Optional[Callable] = None,
        **kwds,
    ):
        super().__init__(**kwds)
        # if resetter_func is None:
        #     raise ValueError("resetter_func parameter is needed")
        self._atol = atol
        self._rtol = rtol
        self._trace = trace
        self._trace_df: Optional[pd.DataFrame] = None
        self._threshold = threshold
        self._resetter_func = resetter_func
        self.inc_pca_wtn: Optional[IncrementalPCA] = None
        self._samples: Optional[Table] = None
        self._samples_flag = False
        self._prev_samples: Optional[Table] = None
        self._prev_samples_flag = False
        self._as_array: Optional[str] = None

    def _proc_as_array(self, data: BaseTable) -> np.ndarray:
        if self._as_array is None:
            if len(data.columns) == 1:
                self._as_array = data.columns[0]
            else:
                self._as_array = ""
        return data[self._as_array].values if self._as_array else data.to_array()

    def create_dependent_modules(self, input_slot: Slot) -> None:
        with self.tagged(self.TAG_DEPENDENT):
            scheduler = self.scheduler()
            with scheduler:
                self.sample = Sample(samples=100, group=self.name, scheduler=scheduler)
                self.sample.input.table = input_slot
                self.input.samples = self.sample.output.select

    def trace_if(self, ret: bool, mean: float, max_: float, len_: int) -> bool:
        if self._trace:
            row: Dict[Union[int, str], Any] = dict(
                Action="RESET" if ret else "PASS", Mean=mean, Max=max_, Length=len_
            )
            if self._trace_df is None:
                self._trace_df = pd.DataFrame(row, index=[0])
            else:
                self._trace_df = self._trace_df.append(row, ignore_index=True)
            if self._trace == "verbose":
                print(row)
        return ret

    def needs_reset(
        self,
        inc_pca: IncrementalPCA,
        inc_pca_wtn: Optional[IncrementalPCA],
        input_table: Table,
        samples: Any,
    ) -> bool:
        if self.has_input_slot("resetter"):
            resetter = self.get_input_slot("resetter")
            resetter.clear_buffers()
            assert self._resetter_func
            if not self._resetter_func(resetter):
                return self.trace_if(False, 0.0, -1.0, len(input_table))
        if self._threshold is not None and len(input_table) >= self._threshold:
            return self.trace_if(False, 0.0, 0.0, len(input_table))
        data = self._proc_as_array(self.filter_columns(input_table, samples))
        assert inc_pca_wtn
        transf_wtn = inc_pca_wtn.transform(data)
        self.maintain_prev_samples(transf_wtn)
        transf_now = inc_pca.transform(data)
        self.maintain_samples(transf_now)
        explained_variance = inc_pca.explained_variance_
        dist = np.sqrt(
            ne.evaluate("((transf_wtn-transf_now)**2)/explained_variance").sum(axis=1)
        )
        _ = explained_variance  # flakes8 does not see variables in numexpr expressions
        mean = np.mean(dist)
        max_ = np.max(dist)
        ret = mean > self._rtol
        return self.trace_if(ret, mean, max_, len(input_table))

    def reset(self) -> None:
        if self.result is not None:
            table = self.result
            assert isinstance(table, Table)
            table.resize(0)

    def starting(self) -> None:
        super().starting()
        samples_slot = self.get_output_slot("samples")
        if samples_slot:
            logger.debug("Maintaining samples")
            self._samples_flag = True
        else:
            logger.debug("Not maintaining samples")
            self._samples_flag = False
        prev_samples_slot = self.get_output_slot("prev_samples")
        if prev_samples_slot:
            logger.debug("Maintaining prev samples")
            self._prev_samples_flag = True
        else:
            logger.debug("Not maintaining prev samples")
            self._prev_samples_flag = False

    def maintain_samples(self, vec: np.ndarray) -> None:
        if not self._samples_flag:
            return
        if isinstance(self._samples, Table):
            self._samples.loc[:, :] = vec
        else:
            df = self._make_df(vec)
            self._samples = Table(
                self.generate_table_name("s_ppca"), data=df, create=True
            )

    def maintain_prev_samples(self, vec: np.ndarray) -> None:
        if not self._prev_samples_flag:
            return
        if isinstance(self._prev_samples, Table):
            self._prev_samples.loc[:, :] = vec
        else:
            df = self._make_df(vec)
            self._prev_samples = Table(
                self.generate_table_name("ps_ppca"), data=df, create=True
            )

    def get_data(self, name: str) -> Any:
        if name == "samples":
            return self._samples
        if name == "prev_samples":
            return self._prev_samples
        return super().get_data(name)

    def _make_df(self, data):
        if self._as_array:
            return {self._as_array: data}
        cols = [f"_pc{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols)

    @process_slot("table", reset_cb="reset")
    @process_slot("samples", reset_if=False)
    @process_slot("transformer", reset_if=False)
    @run_if_any
    def run_step(self, run_number: int, step_size: int, howlong: float):
        """
        """
        assert self.context
        with self.context as ctx:
            input_table = ctx.table.data()
            indices = ctx.table.created.next(length=step_size)
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            transformer = ctx.transformer.data()
            ctx.transformer.clear_buffers()
            inc_pca = transformer.get("inc_pca")
            ctx.samples.clear_buffers()
            if self.inc_pca_wtn is not None:
                samples = ctx.samples.data()
                if self.needs_reset(inc_pca, self.inc_pca_wtn, input_table, samples):
                    self.inc_pca_wtn = None
                    ctx.table.reset()
                    ctx.table.update(run_number)
                    self.reset()
                    indices = ctx.table.created.next(length=step_size)
                    steps = indices_len(indices)
                    if steps == 0:
                        return self._return_run_step(self.state_blocked, steps_run=0)
            else:
                self.inc_pca_wtn = copy.deepcopy(inc_pca)
            data = self._proc_as_array(
                self.filter_columns(input_table, fix_loc(indices))
            )
            reduced = inc_pca.transform(data)
            df = self._make_df(reduced)
            if self.result is None:
                self.result = Table(
                    self.generate_table_name("ppca"), data=df, create=True
                )
            else:
                table = self.result
                assert isinstance(table, Table)
                table.append(df)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)
