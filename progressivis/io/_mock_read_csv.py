import csv

import pandas._libs.lib as lib


def extract_params_docstring(fn, only_defaults=False):
    defaults = fn.__defaults__
    varnames = fn.__code__.co_varnames
    argcount = fn.__code__.co_argcount
    nodefcount = argcount - len(defaults)
    reqargs = ",".join(varnames[0:nodefcount])
    defargs = ",".join(["%s=%s" % (varval[0], repr(varval[1]))
                        for varval in zip(varnames[nodefcount:argcount],
                                          defaults)])
    if only_defaults:
        return defargs
    if not reqargs:
        return defargs
    if not defargs:
        return reqargs
    return reqargs+","+defargs


# Copied from pandas/io/pasers/readers.py
def _mock_read_csv(
    filepath_or_buffer,
    sep=lib.no_default,
    delimiter=None,
    # Column and Index Locations and Names
    header="infer",
    names=lib.no_default,
    index_col=None,
    usecols=None,
    squeeze=None,
    prefix=lib.no_default,
    mangle_dupe_cols=True,
    # General Parsing Configuration
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skipinitialspace=False,
    skiprows=None,
    skipfooter=0,
    nrows=None,
    # NA and Missing Data Handling
    na_values=None,
    keep_default_na=True,
    na_filter=True,
    verbose=False,
    skip_blank_lines=True,
    # Datetime Handling
    parse_dates=None,
    infer_datetime_format=False,
    keep_date_col=False,
    date_parser=None,
    dayfirst=False,
    cache_dates=True,
    # Iteration
    iterator=False,
    chunksize=None,
    # Quoting, Compression, and File Format
    compression="infer",
    thousands=None,
    decimal=".",
    lineterminator=None,
    quotechar='"',
    quoting=csv.QUOTE_MINIMAL,
    doublequote=True,
    escapechar=None,
    comment=None,
    encoding=None,
    encoding_errors="strict",
    dialect=None,
    # Error Handling, remove for backward compatibility
    # error_bad_lines=None,
    # warn_bad_lines=None,
    # TODO(2.0): set on_bad_lines to "error".
    # See _refine_defaults_read comment for why we do this.
    # on_bad_lines=None,
    # Internal
    delim_whitespace=False,
    low_memory=False,
    memory_map=False,
    float_precision=None,
    storage_options=None,
):
    pass


RAW_CSV_DOCSTRING = extract_params_docstring(_mock_read_csv)
