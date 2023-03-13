from __future__ import annotations

import logging

import numpy as np
import copy

from ..core.module import Module, ReturnRunStep, def_input, def_output, def_parameter
from ..core.utils import indices_len, fix_loc
from ..core.pintset import PIntSet
from ..table import BasePTable, PTable, PTableSelectedView
from ..core.decorators import process_slot, run_if_any
from ..utils.psdict import PDict
from . import Sample
import pandas as pd
from sklearn.decomposition import IncrementalPCA  # type: ignore
import numexpr as ne

from typing import Optional, Any, Dict, Union, Callable, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.slot import Slot


logger = logging.getLogger(__name__)


@def_parameter("n_components", np.dtype(int), 2)
@def_input("table", PTable)
@def_output("result", PTableSelectedView)
@def_output("transformer", PDict, required=False)
class PPCA(Module):
    """ """

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        # IncrementalPCA(n_components=self.params.n_components)
        self.inc_pca: Optional[IncrementalPCA] = None
        self.inc_pca_wtn: Optional[IncrementalPCA] = None
        self._as_array: Optional[str] = None

    def predict_step_size(self, duration: float) -> int:
        p = super().predict_step_size(duration)
        return max(p, cast(int, self.params.n_components + 1))

    def reset(self) -> None:
        logger.info("RESET PPCA")
        self.inc_pca = IncrementalPCA(n_components=self.params.n_components)
        self.inc_pca_wtn = None
        if self.result is not None:
            table = self.result
            assert isinstance(table, PTableSelectedView)
            table.selection = PIntSet()

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        """ """
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
                if self.transformer is None:
                    self.transformer = PDict()
                self.transformer["inc_pca"] = self.inc_pca
            self.inc_pca.partial_fit(avs)
            if self.result is None:
                self.result = PTableSelectedView(table, PIntSet(indices))
            else:
                table = self.result
                assert isinstance(table, PTableSelectedView)
                table.selection |= PIntSet(indices)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)

    def create_dependent_modules(
        self,
        atol: float = 0.0,
        rtol: float = 0.001,
        trace: bool = False,
        threshold: Optional[Any] = None,
        resetter: Optional[Any] = None,
        resetter_slot: str = "result",
        resetter_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        with self.grouped():
            s = self.scheduler()
            self.dep.reduced = PPCATransformer(
                scheduler=s,
                atol=atol,
                rtol=rtol,
                trace=trace,
                threshold=threshold,
                resetter_func=resetter_func,
                group=self.name,
            )
            self.dep.reduced.input.table = self.output.result
            self.dep.reduced.input.transformer = self.output.transformer
            if resetter is not None:
                assert callable(resetter_func)
                self.dep.reduced.input.resetter = resetter.output[resetter_slot]
            self.dep.reduced.create_dependent_modules(self.output.result)


@def_input("table", type=PTable, required=True)
@def_input("samples", type=PTable, required=True)
@def_input("transformer", type=PDict, required=True)
@def_input("resetter", type=PDict, required=False)
@def_output("result", PTable)
@def_output("samples", type=PTable, attr_name="_samples", required=False)
@def_output("prev_samples", type=PTable, attr_name="_prev_samples", required=False)
class PPCATransformer(Module):
    """ """

    def __init__(
        self,
        atol: float = 0.0,
        rtol: float = 0.001,
        trace: bool = False,
        threshold: Optional[Any] = None,
        resetter_func: Optional[Callable[..., Any]] = None,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        # if resetter_func is None:
        #     raise ValueError("resetter_func parameter is needed")
        self._atol = atol
        self._rtol = rtol
        self._trace: Union[bool, str] = trace
        self._trace_df: Optional[pd.DataFrame] = None
        self._threshold = threshold
        self._resetter_func = resetter_func
        self.inc_pca_wtn: Optional[IncrementalPCA] = None
        self._samples_flag = False
        self._prev_samples_flag = False
        self._as_array: Optional[str] = None

    def _proc_as_array(self, data: BasePTable) -> np.ndarray[Any, Any]:
        if self._as_array is None:
            if len(data.columns) == 1:
                self._as_array = data.columns[0]
            else:
                self._as_array = ""
        return data[self._as_array].values if self._as_array else data.to_array()

    def create_dependent_modules(self, input_slot: Slot) -> None:
        with self.grouped():
            scheduler = self.scheduler()
            with scheduler:
                self.dep.sample = Sample(
                    samples=100, required="select", group=self.name, scheduler=scheduler
                )
                self.dep.sample.input.table = input_slot
                self.input.samples = self.dep.sample.output.select

    def trace_if(self, ret: bool, mean: float, max_: float, len_: int) -> bool:
        if self._trace:
            row: Dict[Union[int, str], Any] = dict(
                Action="RESET" if ret else "PASS", Mean=mean, Max=max_, Length=len_
            )
            row_df = pd.DataFrame(row, index=[0])
            if self._trace_df is None:
                self._trace_df = row_df
            else:
                self._trace_df = pd.concat([self._trace_df, row_df], ignore_index=True)
            if self._trace == "verbose":
                print(row)
        return ret

    def needs_reset(
        self,
        inc_pca: IncrementalPCA,
        inc_pca_wtn: Optional[IncrementalPCA],
        input_table: PTable,
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
            assert isinstance(table, PTable)
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

    def maintain_samples(self, vec: np.ndarray[Any, Any]) -> None:
        if not self._samples_flag:
            return
        if isinstance(self._samples, PTable):
            self._samples.loc[:, :] = vec
        else:
            df = self._make_df(vec)
            self._samples = PTable(
                self.generate_table_name("s_ppca"), data=df, create=True
            )

    def maintain_prev_samples(self, vec: np.ndarray[Any, Any]) -> None:
        if not self._prev_samples_flag:
            return
        if isinstance(self._prev_samples, PTable):
            self._prev_samples.loc[:, :] = vec
        else:
            df = self._make_df(vec)
            self._prev_samples = PTable(
                self.generate_table_name("ps_ppca"), data=df, create=True
            )

    def _make_df(
        self, data: np.ndarray[Any, Any]
    ) -> Union[Dict[str, np.ndarray[Any, Any]], pd.DataFrame]:
        if self._as_array:
            return {self._as_array: data}
        cols = [f"_pc{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols)

    @process_slot("table", reset_cb="reset")
    @process_slot("samples", reset_if=False)
    @process_slot("transformer", reset_if=False)
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        """ """
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
                self.result = PTable(
                    self.generate_table_name("ppca"), data=df, create=True
                )
            else:
                table = self.result
                assert isinstance(table, PTable)
                table.append(df)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)
