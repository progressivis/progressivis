import os
from contextlib import contextmanager

options = {}
default_values = {}


def get_default_val(pat):
    return default_values.get(pat)


def _get_option(pat, default_val=None):
    return options[pat] if pat in options else default_val


def _set_option(pat, val, default_val=None):
    options[pat] = val
    if default_val is not None:
        default_values[pat] = default_val


def _register_option(pat, val, default_val=None):
    _set_option(pat, val, default_val)

get_option = _get_option
set_option = _set_option
register_option = _register_option


class option_context(object):
    def __init__(self, *args):
        if not (len(args) % 2 == 0 and len(args) >= 2):
            raise ValueError('Need to invoke as'
                             'option_context(pat, val, [(pat, val), ...)).')

        self.ops = list(zip(args[::2], args[1::2]))
        self.undo = None

    def __enter__(self):
        undo = []
        for pat, val in self.ops:
            undo.append((pat, get_option(pat)))

        self.undo = undo

        for pat, val in self.ops:
            set_option(pat, val)

    def __exit__(self, *args):
        if self.undo:
            for pat, val in self.undo:
                set_option(pat, val)


@contextmanager
def config_prefix(prefix):
    global get_option, set_option, register_option

    def wrap(func):
        def inner(key, *args, **kwds):
            pkey = '%s.%s' % (prefix, key)
            return func(pkey, *args, **kwds)

        return inner

    __get_option = get_option
    __set_option = set_option
    __register_option = register_option
    set_option = wrap(set_option)
    get_option = wrap(get_option)
    register_option = wrap(register_option)
    yield None
    set_option = __set_option
    get_option = __get_option
    register_option = __register_option

storage_ = os.getenv('PROGRESSIVIS_STORAGE')
if storage_ is None:
    storage_ = 'numpy'
    
if len(options) == 0:
    register_option('display.precision', 6)
    register_option('display.float_format', None)
    register_option('display.column_space', 12)
    register_option('display.max_rows', 12)
    register_option('display.max_columns', 20)
    register_option('storage.default', storage_)
