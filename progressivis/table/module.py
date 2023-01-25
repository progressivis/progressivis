"""
PTableModule is the base class for all the modules updating internally a PTable
and exposing it as an output slot.
"""
from __future__ import annotations

from ..core.module import Module, ReturnRunStep, JSon  # noqa: F401
from ..core.slot import SlotDescriptor, Slot
from .table_base import PTableSelectedView, BasePTable
from .table import PTable
from .slot_join import SlotJoin
from progressivis.utils import PDict

from typing import Optional, Dict, List, Union, Any, cast


PColumns = Union[None, List[str]]  # , Dict[str, List[str]]]


# pylint: disable=abstract-method
class PModule(Module):
    "Base class for modules managing tables or dicts."
    outputs = []  # #SlotDescriptor("result", type=PTable, required=True)]

    def __init__(
        self,
        columns: Optional[PColumns] = None,
        output_required: Optional[bool] = True,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        if "table_slot" in kwds:
            raise RuntimeError("don't use table_slot")
        if not output_required:
            # Change the descriptor so it's not required any more
            # The original SD is kept in the shared outputs/all_outputs
            # class variables
            sd = SlotDescriptor("result", type=PTable, required=False)
            self.output_descriptors["result"] = sd
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

    def get_first_input_slot(self) -> Optional[str]:
        for k in self.input_slot_names():
            return k
        return None

    def get_columns(
        self, table: Union[BasePTable, Dict[str, Any]], slot: Optional[str] = None
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
        self,
        df: BasePTable,
        indices: Optional[Any] = None,
        slot: Optional[str] = None,
        cols: PColumns = None,
    ) -> BasePTable:
        """
        Return the specified table filtered by the specified indices and
        limited to the columns of interest.
        """
        if self._columns is None or (
            slot is not None and slot not in self._columns_dict
        ):
            if indices is None:
                return df
            return cast(BasePTable, df.loc[indices])
        cols = cols or self.get_columns(df, slot)
        if cols is None:
            return None
        if indices is None:
            if isinstance(cols, (int, str)):
                cols = slice(cols, cols)
            # return df[cols]
        return df.loc[indices, cols]  # type: ignore

    def get_slot_columns(self, name: str) -> List[str]:
        cols = self._columns_dict.get(name)
        if cols is not None:
            return cols
        slot = self.get_input_slot(name)
        if slot is not None:
            return list(slot.data().columns)
        return []

    def has_output_datashape(self, name: str = "table") -> bool:
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
            assert isl is not None
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


class PTableModule(PModule):
    "Base class for modules managing tables."
    outputs = [SlotDescriptor("result", type=PTable, required=True)]

    def __init__(
        self,
        columns: Optional[PColumns] = None,
        output_required: Optional[bool] = True,
        **kwds: Any,
    ) -> None:
        super().__init__(columns, output_required, **kwds)
        self.__result: Union[None, BasePTable] = None

    @property
    def result(self) -> Union[None, BasePTable]:
        return self.__result

    @result.setter
    def result(self, val: Union[BasePTable, None]) -> None:
        if self.__result is not None:
            raise KeyError("result cannot be assigned more than once")
        self.__result = val

    def close_all(self) -> None:
        super().close_all()
        if isinstance(self.__result, PTable) and self.__result.storagegroup is not None:
            self.__result.storagegroup.close_all()

    @property
    def selected(self) -> PTableSelectedView:
        return cast(PTableSelectedView, self.__result)

    def make_slot_join(self, *slots: Slot) -> SlotJoin:
        return SlotJoin(self, *slots)

    def get_data(self, name: str) -> Any:
        if name in ("result", "table"):
            return self.result
        return super().get_data(name)


class PDictModule(PModule):
    "Base class for modules managing dicts."
    outputs = [SlotDescriptor("result", type=PDict, required=True)]

    def __init__(
        self,
        columns: Optional[PColumns] = None,
        output_required: Optional[bool] = True,
        **kwds: Any,
    ) -> None:
        super().__init__(columns, output_required, **kwds)
        self.__result: Union[None, PDict] = None

    @property
    def result(self) -> Union[None, PDict]:
        return self.__result

    @result.setter
    def result(self, val: Union[PDict, None]) -> None:
        if self.__result is not None:
            raise KeyError("result cannot be assigned more than once")
        self.__result = val

    def get_data(self, name: str) -> Any:
        print("name", name)
        if name in ("result", "psdict"):
            return self.result
        return super().get_data(name)
