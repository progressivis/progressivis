# This file comes from https://github.com/blaze/datashape/tree/c9d2bd75414a69d94498e7340ef9dd5fce903007/datashape/error.py
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
"""Error handling"""

syntax_error = """

  File {filename}, line {lineno}
    {line}
    {pointer}

{error}: {msg}
"""

class DataShapeSyntaxError(SyntaxError):
    """
    Makes datashape parse errors look like Python SyntaxError.
    """
    def __init__(self, lexpos, filename, text, msg=None):
        self.lexpos = lexpos
        self.filename = filename
        self.text = text
        self.msg = msg or 'invalid syntax'
        self.lineno = text.count('\n', 0, lexpos) + 1
        # Get the extent of the line with the error
        linestart = text.rfind('\n', 0, lexpos)
        if linestart < 0:
            linestart = 0
        else:
            linestart = linestart + 1
        lineend = text.find('\n', lexpos)
        if lineend < 0:
            lineend = len(text)
        self.line = text[linestart:lineend]
        self.col_offset = lexpos - linestart

    def __str__(self):
        pointer = ' ' * self.col_offset + '^'

        return syntax_error.format(
            filename=self.filename,
            lineno=self.lineno,
            line=self.line,
            pointer=pointer,
            msg=self.msg,
            error=self.__class__.__name__,
        )

    def __repr__(self):
        return str(self)
