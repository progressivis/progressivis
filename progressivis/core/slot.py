"""
Slots between modules.
"""
from __future__ import annotations

from typing import (
    Any,
    Optional,
    Dict,
    Type,
    TYPE_CHECKING,
    Union,
    List,
    Tuple,
)
from dataclasses import dataclass
import copy
import logging
from .changemanager_base import EMPTY_BUFFER, BaseChangeManager

# from .changemanager_literal import LiteralChangeManager

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .module import Module
    from .scheduler import Scheduler
    from .changemanager_base import ChangeBuffer, _accessor

SlotType = Optional[Union[Type[Any], Tuple[Type[Any], ...]]]


class SlotDescriptor:
    "SlotDescriptor is used in modules to describe the input/output slots."
    __slots__ = (
        "name",
        "type",
        "required",
        "multiple",
        "datashape",
        "buffer_created",
        "buffer_updated",
        "buffer_deleted",
        "buffer_exposed",
        "buffer_masked",
        "hint_type",
    )

    def __init__(
        self,
        name: str,
        type: SlotType = None,
        required: bool = True,
        multiple: bool = False,
        datashape: Optional[Dict[str, Union[str, List[str]]]] = None,
        buffer_created: bool = True,
        buffer_updated: bool = True,
        buffer_deleted: bool = True,
        buffer_exposed: bool = True,
        buffer_masked: bool = True,
        hint_type: Any = None,
    ) -> None:
        self.name = name
        self.type = type
        self.required = required
        self.multiple = multiple
        self.datashape = datashape
        self.buffer_created = buffer_created
        self.buffer_updated = buffer_updated
        self.buffer_deleted = buffer_deleted
        self.buffer_exposed = buffer_exposed
        self.buffer_masked = buffer_masked
        self.hint_type = hint_type

    def __str__(self) -> str:
        return (
            f"SlotDescriptor({self.name}, "
            f"type={self.type}, "
            f"required={self.required}, "
            f"multiple={self.multiple})"
        )

    def __repr__(self) -> str:
        return str(self)


@dataclass(kw_only=True)
class SlotHint:
    slot: "Slot"
    hint: Any


class Slot:
    "A Slot manages one connection between two modules."

    def __init__(
        self,
        output_module: Module,
        output_name: str,
        input_module: Optional[Module],
        input_name: Optional[str],
    ):
        self.output_name = output_name
        self.output_module = output_module
        self.input_name = input_name
        self.input_module = input_module
        self.original_name: Optional[str] = None
        self._name: str
        self.changes: Optional[BaseChangeManager] = None
        self.meta: Optional[Any] = None
        self._hint: Optional[Any] = None

    def name(self) -> str:
        "Return the name of the slot"
        assert self.input_module is not None and self.input_name is not None
        if not hasattr(self, "_name"):
            self._name = self.input_module.name + "_" + self.input_name
        return self._name

    def data(self) -> Any:
        "Return the data associated with this slot"
        return self.output_module.get_data(self.output_name)

    def has_data(self) -> bool:
        return self.data() is not None and len(self.data()) > 0

    def scheduler(self) -> Scheduler:
        "Return the scheduler associated with this slot"
        return self.output_module.scheduler()

    def input_descriptor(self) -> SlotDescriptor:
        assert self.input_module is not None and self.input_name is not None
        if self.original_name:
            return self.input_module.input_slot_descriptor(self.original_name)
        return self.input_module.input_slot_descriptor(self.input_name)

    def output_descriptor(self) -> SlotDescriptor:
        return self.output_module.output_slot_descriptor(self.output_name)

    def __str__(self) -> str:
        assert self.input_module is not None
        if self.original_name:
            name = "Slot(%s[%s]->%s[%s/%s])" % (
                self.output_module.name,
                self.output_name,
                self.input_module.name,
                self.input_name,
                self.original_name,
            )
        else:
            name = "Slot(%s[%s]->%s[%s])" % (
                self.output_module.name,
                self.output_name,
                self.input_module.name,
                self.input_name,
            )
        return name

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Slot):
            return False
        return (
            self.output_name == other.output_name
            and self.output_module == other.output_module
            and self.input_name == other.input_name
            and self.input_module == other.input_module
            and self.original_name == other.original_name
        )

    def __neq__(self, other: Any) -> bool:
        return not (self == other)

    @staticmethod
    def compare(self: Slot, other: Slot) -> int:
        if self == other:
            return 0
        if (
            self.output_name < other.output_name
            or self.output_module.name < other.output_module.name
            or (
                self.input_name is not None
                and other.input_name is not None
                and self.input_name < other.input_name
            )
            or (
                self.input_module is not None
                and other.input_module is not None
                and self.input_module.name < other.input_module.name
            )
            or (
                self.original_name is not None
                and other.original_name is not None
                and self.original_name < other.original_name
            )
        ):
            return -1
        return 1

    def last_update(self) -> int:
        "Return the run_number of the last update for this slot"
        assert self.input_module is not None
        if self.changes:
            return self.changes.last_update()
        return self.input_module.last_update()

    def to_json(self) -> Dict[str, Any]:
        """
        Return a dictionary describing this slot, meant to be
        serialized in json.
        """
        assert self.input_module is not None
        return {
            "output_name": self.output_name,
            "output_module": self.output_module.name,
            "input_name": self.input_name,
            "input_module": self.input_module.name,
        }

    def connect(self) -> None:
        "Declares the connection in the Dataflow"
        dataflow = self.output_module.dataflow()  # also in input_module?
        if dataflow:
            dataflow.add_connection(self)

    def validate_types(self) -> bool:
        "Validate the types of the endpoints connected through this slot"
        assert self.input_module is not None and self.input_name is not None
        output_type = self.output_module.output_slot_type(self.output_name)
        input_type = self.input_module.input_slot_type(self.input_name)
        if output_type is None or input_type is None:
            return True
        if isinstance(output_type, tuple):
            for out in output_type:
                if issubclass(out, input_type):
                    return True
        elif issubclass(output_type, input_type):
            return True
        if (
            not isinstance(input_type, type)
            and callable(input_type)
            and input_type(output_type)
        ):
            return True

        logger.error(
            "Incompatible types for slot (%s,%s) in %s",
            input_type,
            output_type,
            str(self),
        )
        return False

    def create_changes(
        self,
        buffer_created: bool = True,
        buffer_updated: bool = False,
        buffer_deleted: bool = False,
        buffer_exposed: bool = False,
        buffer_masked: bool = False,
    ) -> Optional[BaseChangeManager]:
        "Create a ChangeManager associated with the type of the slot's data."
        data = self.data()
        if data is not None:
            return self.create_changemanager(
                type(data),
                self,
                buffer_created=buffer_created,
                buffer_updated=buffer_updated,
                buffer_deleted=buffer_deleted,
                buffer_exposed=buffer_exposed,
                buffer_masked=buffer_masked,
            )
        return None

    def update(
        self,
        run_number: int,
        buffer_created: bool = True,
        buffer_updated: bool = True,
        buffer_deleted: bool = True,
        buffer_exposed: bool = True,
        buffer_masked: bool = True,
        manage_columns: bool = True,
    ) -> None:
        # pylint: disable=too-many-arguments
        "Compute the changes that occur since this slot has been updated."
        if self.changes is None:
            desc = self.input_descriptor()
            # create_changes always return a ChangeManager
            self.changes = self.create_changes(
                buffer_created=desc.buffer_created,
                buffer_updated=desc.buffer_updated,
                buffer_deleted=desc.buffer_deleted,
                buffer_exposed=desc.buffer_exposed,
                buffer_masked=desc.buffer_masked,
            )
        if self.changes:
            df = self.data()
            self.changes.update(run_number, df, self.name())

    def reset(self) -> None:
        "Reset the slot"
        if self.changes:
            self.changes.reset(self.name())

    def clear_buffers(self) -> None:
        "Clear all the buffers"
        if self.changes:
            self.changes.clear()

    def has_buffered(self) -> bool:
        """
        Return True if any of the created/updated/deleted information
        is buffered
        """
        return self.changes.has_buffered() if self.changes else False

    def __getitem__(self, hint: Any) -> SlotHint:
        return SlotHint(slot=self, hint=hint)

    @property
    def created(self) -> ChangeBuffer:
        "Return the buffer for created rows"
        return self.changes.created if self.changes else EMPTY_BUFFER

    @property
    def updated(self) -> ChangeBuffer:
        "Return the buffer for updated rows"
        return self.changes.updated if self.changes else EMPTY_BUFFER

    @property
    def deleted(self) -> ChangeBuffer:
        "Return the buffer for deleted rows"
        return self.changes.deleted if self.changes else EMPTY_BUFFER

    @property
    def base(self) -> _accessor:
        "Return an accessor"
        assert self.changes
        return self.changes.base

    @property
    def selection(self) -> _accessor:
        "Return an accessor"
        assert self.changes
        return self.changes.selection

    @property
    def changemanager(self) -> Optional[BaseChangeManager]:
        "Return the ChangeManager"
        return self.changes

    @property
    def hint(self) -> Any:
        return copy.copy(self._hint)

    changemanager_classes: Dict[Any, Type[BaseChangeManager]] = {}

    @staticmethod
    def create_changemanager(
        datatype: SlotType,
        slot: Slot,
        buffer_created: bool,
        buffer_updated: bool,
        buffer_deleted: bool,
        buffer_exposed: bool,
        buffer_masked: bool,
    ) -> Optional[BaseChangeManager]:
        """
        Create the ChangeManager responsible for this slot type or
        None if no ChangeManager is registered for that type.
        """
        # pylint: disable=too-many-arguments
        logger.debug("create_changemanager(%s, %s)", datatype, slot)
        queue: List[SlotType] = [datatype]
        processed = set()
        while queue:
            datatype = queue.pop()
            if datatype in processed:
                continue
            processed.add(datatype)
            cls = Slot.changemanager_classes.get(datatype)
            if cls is not None:
                logger.info(
                    "Creating changemanager %s for datatype %s" " of slot %s",
                    cls,
                    datatype,
                    slot,
                )
                return cls(
                    slot,
                    buffer_created,
                    buffer_updated,
                    buffer_deleted,
                    buffer_exposed,
                    buffer_masked,
                )
            elif isinstance(datatype, type):  # hasattr(datatype, "__bases__"):
                queue += datatype.__bases__
        logger.info(
            "Creating LiteralChangeManager for datatype %s of slot %s", datatype, slot
        )
        return None

    @staticmethod
    def add_changemanager_type(datatype: Any, cls: Type[BaseChangeManager]) -> None:
        """
        Declare a ChangerManager class for a slot type
        """
        assert isinstance(datatype, type)
        assert isinstance(cls, type)
        Slot.changemanager_classes[datatype] = cls
