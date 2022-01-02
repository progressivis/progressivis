"""
TableModule is the base class for all the modules updating internally a Table
and exposing it as an output slot.
"""
from __future__ import annotations

from ..core.module import Module, ReturnRunStep, JSon  # noqa: F401
from ..core.slot import SlotDescriptor
from .table_base import TableSelectedView
from .table import Table
from .slot_join import SlotJoin
from progressivis.utils import PsDict

from typing import Optional, Dict, List, Union, Any, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.table import BaseTable

Columns = Union[None, List[str]]  # , Dict[str, List[str]]]


# pylint: disable=abstract-method
class TableModule(Module):
    "Base class for modules managing tables."
    outputs = [SlotDescriptor("result", type=Table, required=True)]

    def __init__(self, columns: Columns = None, output_required: bool = True, **kwds):
        super(TableModule, self).__init__(**kwds)
        if "table_slot" in kwds:
            raise RuntimeError("don't use table_slot")
        if not output_required:
            sd = SlotDescriptor("result", type=Table, required=True)
        else:
            sd = SlotDescriptor("result", type=Table, required=False)
        self.outputs.append(sd)
        self.all_outputs.append(sd)

        self._columns: Optional[List[str]] = None
        self._columns_dict: Dict[str, List[str]] = {}
        if isinstance(columns, dict):
            assert len(columns)
            for v in columns.values():
                self._columns = v
                break
            self._columns_dict = columns
        elif isinstance(columns, list):  # backward compatibility
            self._columns = columns
            for k in self._input_slots.keys():
                self._columns_dict[k] = columns
                break
        else:
            assert columns is None
        self.__result: Union[None, BaseTable, PsDict] = None

    def get_first_input_slot(self):
        for k in self._input_slots.keys():
            return k

    def close_all(self) -> None:
        super(TableModule, self).close_all()
        if isinstance(self.__result, Table) and self.__result.storagegroup is not None:
            self.__result.storagegroup.close_all()

    @property
    def result(self) -> Union[None, BaseTable, PsDict]:
        return self.__result

    @result.setter
    def result(self, val: Union[Table, PsDict]):
        if self.__result is not None:
            raise KeyError("result cannot be assigned more than once")
        self.__result = val

    @property
    def table(self) -> Table:
        return cast(Table, self.__result)

    @property
    def psdict(self) -> PsDict:
        return cast(PsDict, self.__result)

    @property
    def selected(self) -> TableSelectedView:
        return cast(TableSelectedView, self.__result)

    def get_data(self, name: str):
        if name in ("result", "table"):
            return self.result
        return super(TableModule, self).get_data(name)

    def get_columns(
        self, table: Union[BaseTable, dict], slot: Optional[str] = None
    ) -> List[str]:
        """
        Return all the columns of interest from the specified table.

        If the module has been created without a list of columns, then all
        the columns of the table are returned.
        Otherwise, the interesection between the specified list and the
        existing columns is returned.
        """
        if table is None:
            return None
        if slot is None:
            _columns = self._columns
        else:
            _columns = self._columns_dict.get(slot)
        df_columns = table.keys() if isinstance(table, dict) else table.columns
        if _columns is None:
            _columns = list(df_columns)
        else:
            cols = set(_columns)
            diff = cols.difference(df_columns)
            for column in diff:
                _columns.remove(column)  # maintain the order
        return _columns

    def filter_columns(
        self, df: BaseTable, indices: Optional[Any] = None, slot: Optional[str] = None
    ) -> BaseTable:
        """
        Return the specified table filtered by the specified indices and
        limited to the columns of interest.
        """
        if self._columns is None or (
            slot is not None and slot not in self._columns_dict
        ):
            if indices is None:
                return df
            return df.loc[indices]
        cols = self.get_columns(df, slot)
        if cols is None:
            return None
        if indices is None:
            if isinstance(cols, (int, str)):
                cols = slice(cols, cols)
            # return df[cols]
        return df.loc[indices, cols]

    def get_slot_columns(self, name: str) -> List[str]:
        cols = self._columns_dict.get(name)
        if cols is None:
            cols = self.get_input_slot(name).data().columns
        return cols

    def has_output_datashape(self, name="table") -> bool:
        for osl in self.all_outputs:
            if osl.name == name:
                break
        else:
            raise ValueError("Output slot not declared")
        return osl.datashape is not None

    def get_output_datashape(self, name: str = "table") -> str:
        for osl in self.all_outputs:
            if osl.name == name:
                # output_ = osl
                break
        else:
            raise ValueError("Output slot not declared")
        if osl.datashape is None:
            raise ValueError("datashape not defined on {} output slot")
        dshapes = []
        for k, v in osl.datashape.items():
            isl = self.get_input_slot(k)
            table = isl.data()
            if v == "#columns":
                colsn = self.get_slot_columns(k)
            elif v == "#all":
                colsn = table._columns
            else:
                assert isinstance(v, list)
                colsn = v
            for colname in colsn:
                col = table._column(colname)
                if len(col.shape) > 1:
                    dshapes.append(f"{col.name}: {col.shape[1]} * {col.dshape}")
                else:
                    dshapes.append(f"{col.name}: {col.dshape}")
        return "{" + ",".join(dshapes) + "}"

    def get_datashape_from_expr(self) -> str:
        if hasattr(self, "expr"):
            expr = getattr(self, "expr")
            return "{{{cols}}}".format(cols=",".join(expr.keys()))
        raise ValueError("expr attribute not defined")

    def make_slot_join(self, *slots):
        return SlotJoin(self, *slots)
