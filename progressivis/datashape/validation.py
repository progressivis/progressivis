# This file comes from https://github.com/blaze/datashape/tree/c9d2bd75414a69d94498e7340ef9dd5fce903007/datashape/validation.py
#
# It is licensed under the following license:
#
# Copyright (c) 2012, Continuum Analytics, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# type: ignore
# flake8: noqa
# -*- coding: utf-8 -*-

"""
Datashape validation.
"""

from . import coretypes as T


def traverse(f, t):
    """
    Map `f` over `t`, calling `f` with type `t` and the map result of the
    mapping `f` over `t` 's parameters.

    Parameters
    ----------
    f : callable
    t : DataShape

    Returns
    -------
    DataShape
    """
    if isinstance(t, T.Mono) and not isinstance(t, T.Unit):
        return f(t, [traverse(f, p) for p in t.parameters])
    return t


def validate(ds):
    """
    Validate a datashape to see whether it is well-formed.

    Parameters
    ----------
    ds : DataShape

    Examples
    --------
    >>> from datashape import dshape
    >>> dshape('10 * int32')
    dshape("10 * int32")
    >>> dshape('... * int32')
    dshape("... * int32")
    >>> dshape('... * ... * int32') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: Can only use a single wildcard
    >>> dshape('T * ... * X * ... * X') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: Can only use a single wildcard
    >>> dshape('T * ...') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DataShapeSyntaxError: Expected a dtype
    """
    traverse(_validate, ds)


def _validate(ds, params):
    if isinstance(ds, T.DataShape):
        # Check ellipses
        ellipses = [x for x in ds.parameters if isinstance(x, T.Ellipsis)]
        if len(ellipses) > 1:
            raise TypeError("Can only use a single wildcard")
        elif isinstance(ds.parameters[-1], T.Ellipsis):
            raise TypeError("Measure may not be an Ellipsis (...)")
