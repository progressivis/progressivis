from . import ProgressiveTest


from progressivis.table.table import Table
from progressivis.table.row import Row

class TestRow(ProgressiveTest):
    def test_row(self):
        table = Table('table', data={'a': [ 1, 2, 3], 'b': [10.1, 0.2, 0.3]}, create=True)

        row = Row(table)

        self.assertEqual(len(row), 2) # 2 values
        self.assertEqual(row['a'], 3)
        self.assertEqual(row['b'], 0.3)

        row['a'] = 4
        self.assertEqual(row['a'], 4)
        self.assertEqual(table.at[len(table)-1, 'a'], 4)

        table.append({'a': [4, 5], 'b': [0.4, 0.5]})
        self.assertEqual(len(row), 2) # 2 values
        self.assertEqual(row['a'], 5)
        self.assertEqual(row['b'], 0.5)

