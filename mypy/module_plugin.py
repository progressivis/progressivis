from mypy.plugin import Plugin, ClassDefContext
from importlib import import_module
from sqlmypy import add_var_to_class, CB  # type: ignore
from mypy.nodes import TypeInfo
from mypy.types import Instance, UnionType, NoneTyp
from typing import Any
import sys
import os
root_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.insert(1, root_dir)  # in order to import progressivis tests


def decl_deco_hook(ctx: ClassDefContext) -> None:
    try:
        pymod = import_module(".".join(ctx.cls.fullname.split(".")[:-1]))
    except ModuleNotFoundError:  # e.g. tests.test_NN_name.TestClass
        return
    module = pymod.__dict__[ctx.cls.name]
    for attr in module.output_attrs.values():
        if attr not in module.output_types:
            continue  # when custom_attr is True
        typ = module.output_types[attr]
        if typ is not None:
            fullname = f"{typ.__module__}.{typ.__name__}"
            sym = ctx.api.lookup_fully_qualified_or_none(fullname)
            if sym is None:
                print(f"{fullname} not found")
                return
            if not isinstance(sym.node, TypeInfo):
                print(f"{sym.node} is not a TypeInfo")
            typ = Instance(sym.node, [])
            typ = UnionType([typ, NoneTyp()])
        add_var_to_class(attr, typ, ctx.cls.info)


class ModulePlugin(Plugin):
    def get_class_decorator_hook(self, fullname: str) -> CB[ClassDefContext]:
        if fullname == "progressivis.core.module.def_output":
            return decl_deco_hook


def plugin(version: str) -> Any:
    return ModulePlugin
