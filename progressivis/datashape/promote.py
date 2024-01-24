# This file comes from https://github.com/blaze/datashape/tree/c9d2bd75414a69d94498e7340ef9dd5fce903007/datashape/promote.py
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
from __future__ import absolute_import

import numpy as np
import progressivis.datashape


__all__ = ['promote', 'optionify']


def promote(lhs, rhs, promote_option=True):
    """Promote two scalar dshapes to a possibly larger, but compatible type.

    Examples
    --------
    >>> from datashape import int32, int64, Option, string
    >>> x = Option(int32)
    >>> y = int64
    >>> promote(x, y)
    Option(ty=ctype("int64"))
    >>> promote(int64, int64)
    ctype("int64")

    Don't promote to option types.
    >>> promote(x, y, promote_option=False)
    ctype("int64")

    Strings are handled differently than NumPy, which promotes to ctype("object")
    >>> x = string
    >>> y = Option(string)
    >>> promote(x, y) == promote(y, x) == Option(string)
    True
    >>> promote(x, y, promote_option=False)
    ctype("string")

    Notes
    ----
    Except for ``datashape.string`` types, this uses ``numpy.result_type`` for
    type promotion logic.  See the numpy documentation at:

    http://docs.scipy.org/doc/numpy/reference/generated/numpy.result_type.html
    """
    if lhs == rhs:
        return lhs
    left, right = getattr(lhs, 'ty', lhs), getattr(rhs, 'ty', rhs)
    if left == right == datashape.string:
        # Special case string promotion, since numpy promotes to `object`.
        dtype = datashape.string
    else:
        np_res_type = np.result_type(datashape.to_numpy_dtype(left),
                                     datashape.to_numpy_dtype(right))
        dtype = datashape.CType.from_numpy_dtype(np_res_type)
    if promote_option:
        dtype = optionify(lhs, rhs, dtype)
    return dtype


def optionify(lhs, rhs, dshape):
    """Check whether a binary operation's dshape came from
    :class:`~datashape.coretypes.Option` typed operands and construct an
    :class:`~datashape.coretypes.Option` type accordingly.

    Examples
    --------
    >>> from datashape import int32, int64, Option
    >>> x = Option(int32)
    >>> x
    Option(ty=ctype("int32"))
    >>> y = int64
    >>> y
    ctype("int64")
    >>> optionify(x, y, int64)
    Option(ty=ctype("int64"))
    """
    if hasattr(dshape.measure, 'ty'):
        return dshape
    if hasattr(lhs, 'ty') or hasattr(rhs, 'ty'):
        return datashape.Option(dshape)
    return dshape
