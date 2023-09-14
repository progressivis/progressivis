from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from ..core.docstrings import FILENAMES_DOC, RESULT_DOC
from ..utils.errors import ProgressiveError, ProgressiveStopIteration
from ..utils.inspect import filter_kwds, extract_params_docstring
from ..core.module import Module
from ..table.table import PTable
from ..core.module import def_input, def_output, document
from ..table.dshape import (
    dshape_from_dataframe,
    array_dshape,
    dshape_from_dict,
)
from progressivis.core.utils import force_valid_id_columns
from progressivis.io.read_csv import read_csv, recovery, is_recoverable, InputSource

from typing import (
    Any,
    Union,
    Optional,
    Dict,
    Tuple,
    List,
    Callable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from progressivis.core.module import ModuleState, ReturnRunStep
    from progressivis.io.read_csv import Parser
    from progressivis.table.dshape import DataShape


logger = logging.getLogger(__name__)


@document
@def_input("filenames", PTable, required=False, doc=FILENAMES_DOC)
@def_output("result", PTable, doc=RESULT_DOC)
class CSVLoader(Module):
    """
    Warning : this module do not wait for "filenames"
    """

    def __init__(
        self,
        filepath_or_buffer: Optional[Any] = None,
        filter_: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        force_valid_ids: bool = True,
        fillvalues: Optional[Dict[str, Any]] = None,
        as_array: Optional[str | Dict[str, List[str]] | Callable[[List[str]], Dict[str, List[str]]]] = None,
        timeout: Optional[float] = None,
        save_context: bool = True,
        recovery: bool = False,
        recovery_tag: Union[str, int] = "",
        recovery_table_size: int = 3,
        save_step_size: int = 100000,
        **kwds: Any,
    ) -> None:
        r"""
        Args:
            filepath_or_buffer: str, path object or file-like object accepted by :func:`pandas.read_csv`

                NB: ``filenames`` slot and filepath_or_buffer init parameter  cannot both be defined
            filter\_: filtering function to be applied on input data at loading time

                Example:
                    >>> def filter_(df):
                    ...     lon = df['dropoff_longitude']
                    ...     lat = df['dropoff_latitude']
                    ...     return df[(lon>-74.10)&(lon<-73.7)&(lat>40.60)&(lat<41)]
            force_valid_ids: force renaming of columns to make their names valid identifiers according to the `language definition  <https://docs.python.org/3/reference/lexical_analysis.html#identifiers>`_
            fillvalues: the default values of the columns specified as a dictionary (see :class:`PTable <progressivis.table.PTable>`)
            as_array: allows lists of *n* input columns to be grouped into output *n-D* columns (grouped column types must be identical):

                * when `as_array` is a ``Dict[str, List[str]]`` every entry represents a grouping rule

                    Example:
                        >>> CSVLoader(
                        ...             get_dataset("bigfile"),
                        ...             index_col=False,
                        ...             as_array={
                        ...                 "first_group": ["_1", "_2", "_3"],
                        ...                 "second_group": ["_11", "_12", "_13"],
                        ...             },
                        ...             header=None,
                        ...             scheduler=s,
                        ...         )

                * when `as_array` is a ``str`` it gives the name of an unique *n-D* output column which groups all input columns
                * when `as_array` is a ``Callable`` it must be able to process the list of all input columns and return a ``Dict[str, List[str]]``

                    Example:
                        >>> CSVLoader(
                        ...                 get_dataset("mnist_784"),
                        ...                 index_col=False,
                        ...                 as_array=lambda cols: {"array": [
                        ...                     c for c in cols if c != "class"]},
                        ...                 scheduler=s,
                        ...             )

            timeout: stop waiting for a response and raise an exception after a given number of seconds
            save_context: saving  information allowing recovery after a failure if ``True`` and if ``filepath_or_buffer``
                 designates a recoverable support (i.e. a filepath)

                NB: currently backup only works when data input is provided via ``filepath_or_buffer``
            recovery: when ``False`` (default) the data entry is read from the beginning
                 otherwise the previous reading is resumed using the save information
            recovery_tag: customize save tags (useful mainly for debugging)
            recovery_table_size: defines the size of the recovery table (useful mainly for debugging),
            save_step_size: defines de number of rows to read between two context snapshots,
            kwds: extra keyword args to be passed to :func:`pandas.read_csv` and :class:`Module <progressivis.core.Module>` superclass
        """
        super(CSVLoader, self).__init__(**kwds)
        self.tags.add(self.TAG_SOURCE)
        self.default_step_size = kwds.get("chunksize", 1000)  # initial guess
        kwds.setdefault("chunksize", self.default_step_size)
        # Filter out the module keywords from the csv loader keywords
        csv_kwds = filter_kwds(kwds, pd.read_csv)
        # When called with a specified chunksize, it returns a parser
        self.filepath_or_buffer = filepath_or_buffer
        self.force_valid_ids = force_valid_ids
        self.parser: Optional[Parser] = None
        self.csv_kwds = csv_kwds
        self._compression = csv_kwds.get("compression", "infer")
        csv_kwds["compression"] = None
        self._encoding = csv_kwds.get("encoding", None)
        csv_kwds["encoding"] = None
        self._rows_read = 0
        if filter_ is not None and not callable(filter_):
            raise ProgressiveError("filter parameter should be callable or None")
        self._filter = filter_
        # self._input_stream: Optional[Any] = (
        #     None  # stream that returns a position through the 'tell()' method
        # )
        self._input_encoding = None
        self._input_compression = None
        self._input_size = 0  # length of the file or input stream when available
        self._timeout_csv = timeout
        self._table_params: Dict[str, Any] = dict(name=self.name, fillvalues=fillvalues)
        self._as_array = as_array
        self._save_context = (
            True
            if save_context and is_recoverable(filepath_or_buffer)
            else False
        )
        self._recovery = recovery
        self._recovery_table_size = recovery_table_size
        self._recovery_table: Optional[PTable] = None
        self._recovery_table_name = f"csv_loader_recovery_{recovery_tag}"
        self._recovery_table_inv: Optional[PTable] = None
        self._recovery_table_inv_name = f"csv_loader_recovery_invariant_{recovery_tag}"
        self._save_step_size = save_step_size
        self._last_saved_id = 0
        if self._recovery and not self.recovery_tables_exist():
            self._recovery = False
        if not self._recovery:
            self.trunc_recovery_tables()

    def recovery_tables_exist(self) -> bool:
        try:
            PTable(name=self._recovery_table_name, create=False)
        except ValueError as ve:
            if "exist" in ve.args[0]:
                print("WARNING: recovery table does not exist")
                return False
            raise
        try:
            PTable(name=self._recovery_table_inv_name, create=False)
        except Exception as ve:
            # FIXME JDF: is that the right way?
            if "exist" in ve.args[0]:  # FIXME
                print("WARNING: recovery table invariant does not exist")
                return False
            raise
        return True

    def trunc_recovery_tables(self) -> None:
        len_ = 0
        rt: Optional[PTable] = None
        try:
            rt = PTable(name=self._recovery_table_name, create=False)
            len_ = len(rt)
        except Exception:
            pass
        if len_ and rt is not None:
            rt.truncate()
        len_ = 0
        try:
            rt = PTable(name=self._recovery_table_inv_name, create=False)
            len_ = len(rt)
        except Exception:
            pass
        if len_ and rt is not None:
            rt.truncate()

    def rows_read(self) -> int:
        "Return the number of rows read so far."
        return self._rows_read

    def is_ready(self) -> bool:
        if self.has_input_slot("filenames"):
            fn = self.get_input_slot("filenames")
            if fn.created is None or fn.created.any():
                return True
        return super(CSVLoader, self).is_ready()

    def is_data_input(self) -> bool:
        # pylint: disable=no-self-use
        "Return True if this module brings new data"
        return True

    def create_input_source(self, filepath: str) -> InputSource:
        usecols = self.csv_kwds.get("usecols")
        return InputSource.create(
            filepath,
            encoding=self._encoding,
            compression=self._compression,
            timeout=self._timeout_csv,
            start_byte=0,
            usecols=usecols,
        )

    def close(self) -> None:
        # if self._input_stream is None:
        #     return
        # try:
        #     self._input_stream.close()
        #     # pylint: disable=bare-except
        # except Exception:
        #     pass
        # self._input_stream = None
        self._input_encoding = None
        self._input_compression = None
        self._input_size = 0

    def get_progress(self) -> Tuple[int, int]:
        if self._input_size == 0:
            return (0, 0)
        pos = 0  # self._input_stream.tell()
        return (pos, self._input_size)

    def validate_parser(self, run_number: int) -> ModuleState:
        if self.parser is None:
            if self.filepath_or_buffer is not None and self.has_input_slot("filenames"):
                raise ProgressiveError("'filepath_or_buffer' parameter and"
                                       " 'filenames' slot cannot both be defined")
            if self.filepath_or_buffer is not None:
                if not self._recovery:
                    try:
                        self.parser = read_csv(
                            self.create_input_source(self.filepath_or_buffer),
                            **self.csv_kwds,
                        )
                    except IOError as e:
                        logger.error(
                            "Cannot open file %s: %s", self.filepath_or_buffer, e
                        )
                        self.parser = None
                        return self.state_terminated
                    self.filepath_or_buffer = None
                else:  # do recovery
                    try:
                        if self._recovery_table is None:
                            self._recovery_table = PTable(
                                name=self._recovery_table_name, create=False
                            )
                        if self._recovery_table_inv is None:
                            self._recovery_table_inv = PTable(
                                name=self._recovery_table_inv_name, create=False
                            )
                        if self.result is None:
                            self._table_params["name"] = self._recovery_table_inv[
                                "table_name"
                            ].loc[0]
                            self._table_params["create"] = False
                            table = PTable(**self._table_params)
                            self.result = table
                            table.last_id
                    except Exception as e:  # TODO: specify the exception?
                        logger.error(f"Cannot acces recovery table {e}")
                        return self.state_terminated
                    table = self.result
                    try:
                        last_ = self._recovery_table.eval(
                            "last_id=={}".format(len(table)), as_slice=False
                        )
                        len_last = len(last_)
                        if len_last > 1:
                            logger.error("Inconsistent recovery table")
                            return self.state_terminated
                        # last_ = self._recovery_table.argmax()['offset']
                        snapshot: Optional[Dict[str, Any]] = None
                        if len_last == 1:
                            row = self._recovery_table.row(last_[0])
                            assert row is not None
                            snapshot = row.to_dict(ordered=True)
                            if not check_snapshot(snapshot):
                                snapshot = None
                        if (
                            snapshot is None
                        ):  # i.e. snapshot not yet found or inconsistent
                            max_ = -1
                            for i in self._recovery_table.eval(
                                "last_id<{}".format(len(table)), as_slice=False
                            ):
                                row = self._recovery_table.row(i)
                                assert row is not None
                                sn: Dict[str, Any] = row.to_dict(ordered=True)
                                if check_snapshot(sn) and sn["last_id"] > max_:
                                    max_, snapshot = sn["last_id"], sn
                            if max_ < 0:
                                # logger.error('Cannot acces recovery table (max_<0)')
                                return self.state_terminated
                            table.drop(slice(max_ + 1, None, None), truncate=True)
                        assert snapshot
                        self._recovered_csv_table_name = snapshot["table_name"]
                    except Exception as e:
                        logger.error("Cannot read the snapshot %s", e)
                        return self.state_terminated
                    try:
                        self.parser = recovery(
                            snapshot, self.filepath_or_buffer, **self.csv_kwds
                        )
                    except Exception as e:
                        logger.error("Cannot recover from snapshot %s, %s", snapshot, e)
                        self.parser = None
                        return self.state_terminated
                    self.filepath_or_buffer = None

            else:  # this case does not support recovery
                fn_slot = None
                if self.has_input_slot("filenames"):
                    fn_slot = self.get_input_slot("filenames")
                if fn_slot is None or fn_slot.output_module is None:
                    return self.state_terminated
                # fn_slot.update(run_number)
                if fn_slot.deleted.any() or fn_slot.updated.any():
                    raise ProgressiveError("Cannot handle input file changes")
                df = fn_slot.data()
                while self.parser is None:
                    indices = fn_slot.created.next(length=1)
                    assert isinstance(indices, slice)
                    if indices.stop == indices.start:
                        return self.state_blocked
                    filename = df.at[indices.start, "filename"]
                    try:
                        self.parser = read_csv(
                            self.create_input_source(filename), **self.csv_kwds
                        )
                    except IOError as e:
                        logger.error("Cannot open file %s: %s", filename, e)
                        self.parser = None
                        # fall through
        return self.state_ready

    def _data_as_array(self, df: pd.DataFrame) -> Tuple[Any, DataShape]:
        if not self._as_array:
            return (df, dshape_from_dataframe(df))
        if callable(self._as_array):
            self._as_array = self._as_array(list(df.columns))  # FIXME
        if isinstance(self._as_array, str):
            data = df.values
            dshape = array_dshape(data, self._as_array)
            return ({self._as_array: data}, dshape)
        if not isinstance(self._as_array, dict):
            raise ValueError(
                f"Unexpected parameter specified to as_array: {self._as_array}"
            )
        columns = set(df.columns)
        ret: Dict[str, pd.api.extensions.ExtensionArray | np.ndarray[Any, Any]] = {}
        for colname, cols in self._as_array.items():
            if colname in ret:
                raise KeyError(f"Duplicate column {colname} in as_array")
            colset = set(cols)
            assert colset.issubset(columns)
            columns -= colset
            view = df[cols]
            values = view.values
            ret[colname] = values
        for colname in columns:
            if colname in ret:
                raise KeyError(f"Duplicate column {colname} in as_array")
            ret[colname] = df[colname].values
        return ret, dshape_from_dict(ret)

    def _needs_save(self) -> bool:
        table = self.result
        if table is None:
            return False
        return table.last_id >= self._last_saved_id + self._save_step_size

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if step_size == 0:  # bug
            logger.error("Received a step_size of 0")
            return self._return_run_step(self.state_ready, steps_run=0)
        status = self.validate_parser(run_number)
        if status == self.state_terminated:
            raise ProgressiveStopIteration("no more filenames")
        elif status == self.state_blocked:
            return self._return_run_step(status, steps_run=0)
        elif status != self.state_ready:
            logger.error("Invalid state returned by validate_parser: %d", status)
            self.close()
            raise ProgressiveStopIteration("Unexpected situation")
        logger.info("loading %d lines", step_size)
        needs_save = self._needs_save()
        assert self.parser
        df_list: List[pd.DataFrame]
        try:
            df_list = self.parser.read(
                step_size, flush=needs_save
            )  # raises StopIteration at EOF
            if not df_list:
                raise ProgressiveStopIteration
        except ProgressiveStopIteration:
            self.close()
            if self.has_input_slot("filenames"):
                fn_slot = self.get_input_slot("filenames")
                assert fn_slot.output_module is not None
            self.parser = None
            return self._return_run_step(self.state_ready, 0)
        df_len = sum([len(df) for df in df_list])
        creates = df_len
        if creates == 0:  # should not happen
            logger.error("Received 0 elements")
            raise ProgressiveStopIteration
        if self._filter is not None:
            df_list = [self._filter(df) for df in df_list]
        creates = sum([len(df) for df in df_list])
        if creates == 0:
            logger.info("frame has been filtered out")
        else:
            self._rows_read += creates
            logger.info("Loaded %d lines", self._rows_read)
            if self.force_valid_ids:
                for df in df_list:
                    force_valid_id_columns(df)
            if self.result is None:
                table = self.result
                data, dshape = self._data_as_array(pd.concat(df_list))
                if not self._recovery:
                    self._table_params["name"] = self.generate_table_name("table")
                    self._table_params["data"] = data
                    self._table_params["dshape"] = dshape
                    self._table_params["create"] = True
                    self.result = PTable(**self._table_params)
                else:
                    self._table_params["name"] = self._recovered_csv_table_name
                    # self._table_params['dshape'] = dshape
                    self._table_params["create"] = False
                    table = PTable(**self._table_params)
                    self.result = table
                    table.append(self._data_as_array(pd.concat(df_list)))
            else:
                table = self.result
                for df in df_list:
                    data, dshape = self._data_as_array(df)
                    table.append(data)
            if (
                self.parser.is_flushed()
                and needs_save
                and self._recovery_table is None
                and self._save_context
            ):
                table = self.result
                snapshot = self.parser.get_snapshot(
                    run_number=run_number,
                    table_name=table.name,
                    last_id=table.last_id,
                )
                self._recovery_table = PTable(
                    name=self._recovery_table_name,
                    data=pd.DataFrame(snapshot, index=[0]),
                    create=True,
                )
                self._recovery_table_inv = PTable(
                    name=self._recovery_table_inv_name,
                    data=pd.DataFrame(
                        dict(
                            table_name=table.name,
                            csv_input=self.filepath_or_buffer,
                        ),
                        index=[0],
                    ),
                    create=True,
                )
                self._last_saved_id = table.last_id
            elif self.parser.is_flushed() and needs_save and self._save_context:
                assert table is not None
                snapshot = self.parser.get_snapshot(
                    run_number=run_number,
                    last_id=table.last_id,
                    table_name=table.name,
                )
                assert self._recovery_table
                self._recovery_table.add(snapshot)
                if len(self._recovery_table) > self._recovery_table_size:
                    oldest = self._recovery_table.argmin()["offset"]
                    self._recovery_table.drop(oldest)
                self._last_saved_id = table.last_id
        return self._return_run_step(self.state_ready, steps_run=creates)


def check_snapshot(snapshot: Dict[str, Any]) -> bool:
    if "check" not in snapshot:
        return False
    hcode = snapshot["check"]
    del snapshot["check"]
    h = hash(tuple(snapshot.values()))
    return bool(h == hcode)


CSV_DOCSTRING = (
    "CSVLoader("
    + extract_params_docstring(pd.read_csv)
    + ","
    + extract_params_docstring(CSVLoader.__init__, only_defaults=True)
    + ",force_valid_ids=False,id=None,tracer=None,predictor=None,storage=None)"
)

try:
    if not CSVLoader.doc_building():
        CSVLoader.__init__.__func__.__doc__ = CSV_DOCSTRING  # type: ignore
except AttributeError:
    try:
        CSVLoader.__init__.__doc__ = CSV_DOCSTRING
    except AttributeError:
        logger.warning("Cannot set CSVLoader docstring")
