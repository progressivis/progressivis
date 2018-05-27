"""
Manage changes in table columns
"""
from __future__ import absolute_import, division, print_function

from collections import namedtuple

ColumnUpdate = namedtuple('ColumnUpdate', ['created', 'updated', 'deleted'])
