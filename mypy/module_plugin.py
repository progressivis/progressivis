from mypy.plugin import Plugin, ClassDefContext
from sqlmypy import add_var_to_class, CB  # type: ignore
from mypy.nodes import TypeInfo, NameExpr, CallExpr
from mypy.types import Instance, UnionType, NoneTyp
from typing import Any, Union, cast


def get_full_name(name):
    if "." in name:
        name = name.split(".")[:-1]
    fullnames = dict(
        PTable="progressivis.table.table.PTable",
        PTableSelectedView="progressivis.table.table_base.PTableSelectedView",
        PColumn="progressivis.table.table.PColumn",
        PDict="progressivis.utils.psdict.PDict"
    )
    return fullnames.get(name)


def get_content(obj):
    if isinstance(obj, NameExpr):
        return obj.name
    return obj.value


def decl_deco_hook(ctx: ClassDefContext) -> None:
    reason: CallExpr = cast(CallExpr, ctx.reason)
    if "custom_attr" in reason.arg_names:
        i = reason.arg_names.index("custom_attr")
        if get_content(reason.args[i]) == "True":
            return
    if "attr_name" in reason.arg_names:
        i = reason.arg_names.index("attr_name")
        attr = get_content(reason.args[i])
    elif "name" in reason.arg_names:
        i = reason.arg_names.index("name")
        attr = get_content(reason.args[i])
    else:
        if not len(reason.args):
            return  # fail
        if reason.arg_kinds[0].name != 'ARG_POS':
            return  # fail
        attr = get_content(reason.args[0])
    typ: Union[Instance, UnionType, NoneTyp, None] = None
    if "type" in reason.arg_names:
        i = reason.arg_names.index("type")
        typ = get_content(reason.args[i])
    else:
        if not len(reason.args):
            return  # fail
        if reason.arg_kinds[1].name == 'ARG_POS':
            typ = get_content(reason.args[1])
    if typ is not None:
        fullname = get_full_name(typ)
        if fullname is None:
            return
        sym = ctx.api.lookup_fully_qualified_or_none(fullname)
        if sym is None:
            print(f"{fullname} not found")
            return
        if not isinstance(sym.node, TypeInfo):
            print(f"{sym.node} is not a TypeInfo")
        else:
            typ = Instance(sym.node, [])
            typ = UnionType([typ, NoneTyp()])
    add_var_to_class(attr, typ, ctx.cls.info)


class ModulePlugin(Plugin):
    def get_class_decorator_hook(self, fullname: str) -> CB[ClassDefContext]:
        if fullname == "progressivis.core.module.def_output":
            return decl_deco_hook


def plugin(version: str) -> Any:
    return ModulePlugin
