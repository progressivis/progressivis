from inspect import signature, Parameter


from typing import Callable, Dict, Any


def extract_params_docstring(fn: Callable[..., Any], only_defaults: bool = False) -> str:
    sig = signature(fn)
    par = [(p.name, p.default)
           for p in sig.parameters.values()]

    def_only = ",".join([f"{name}={repr(default)}"
                         for (name, default) in par
                         if default is not Parameter.empty])
    if only_defaults:
        return def_only
    return ",".join([f"{name}"
                     for (name, default) in par
                     if default is Parameter.empty])+","+def_only


def filter_kwds(kwds: Dict[str, Any], function_or_method: Callable[..., Any]) -> Dict[str, Any]:
    par = signature(function_or_method).parameters
    return {k: kwds[k] for k in kwds.keys() & par.keys()
            if par[k].default is not Parameter.empty}
