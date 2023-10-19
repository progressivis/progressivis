"Hub of Module Slots"

from __future__ import annotations

from typing import (
    Iterable,
    List,
)
from dataclasses import dataclass

from progressivis import Slot, ProgressiveError, Module


@dataclass
class SlotProxy:
    output_name: str
    output_module: Module | SlotHub


class SlotHub:
    """
    Presents a unified facade of output slots for a list of interconnected modules.
    """

    def __init__(self) -> None:
        self.output = OutputProxies(self)
        self._output_slots: dict[str, SlotProxy] = {}

    def get(self, name: str) -> SlotProxy | None:
        return self._output_slots[name]

    def has_output_slot(self, name: str) -> bool:
        return name in self._output_slots

    def output_slot_names(self) -> List[str]:
        return list(self._output_slots.keys())

    def create_slot_module(self, name: str) -> SlotProxy:
        raise ProgressiveError(f"Cannot create module for slot {name}")

    def add_proxy(
        self, name: str, output_name: str, output_module: Module | SlotHub
    ) -> None:
        if name in self._output_slots:
            raise ProgressiveError(f"Output slot '{name}' already registered in Hub")
        if output_module and not output_module.has_output_slot(output_name):
            raise ProgressiveError(
                f"No output slot named '{output_name}' in {output_module}"
            )
        self._output_slots[name] = SlotProxy(output_name, output_module)


class OutputProxies:
    """
    Convenience class to refer to output slots by name
    as if they were attributes.
    """

    def __init__(self, hub: SlotHub):
        self.hub: SlotHub
        self.__dict__["hub"] = hub

    def __setattr__(self, name: str, slot: Slot) -> None:
        raise ProgressiveError("Output slots cannot be assigned, only read")

    def __getattr__(self, name: str) -> Slot:
        while True:
            proxy = self.hub.get(name)
            if proxy is None:  # should create the module before passing its slot
                proxy = self.hub.create_slot_module(name)
                self.hub.add_proxy(name, proxy.output_name, proxy.output_module)
            if isinstance(proxy.output_module, Module):  # Follow if SlotHub again
                break
            name = proxy.output_name
        assert isinstance(proxy.output_module, Module)
        return proxy.output_module.create_slot(proxy.output_name, None, None)

    def __getitem__(self, name: str) -> Slot:
        return self.__getattr__(name)

    def __dir__(self) -> Iterable[str]:
        return self.hub.output_slot_names()
