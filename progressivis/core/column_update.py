"""
Manage changes in table columns
"""

from collections import namedtuple

ColumnUpdate = namedtuple('ColumnUpdate', ['created', 'updated', 'deleted'])
