import tempfile
import os
import os.path
import shutil

from progressivis.core.utils import RandomBytesIO
from . import ProgressiveTest


class TestRandomCsv(ProgressiveTest):
    def setUp(self):
        # fixed rows number
        self.dtemp = tempfile.mkdtemp(prefix="p10s_")
        fixed_rows_obj = RandomBytesIO(cols=10, rows=50)
        self.fixed_rows_file = os.path.join(self.dtemp, "fixed_rows.csv")
        fixed_rows_obj.save(self.fixed_rows_file)
        # fixed size
        fixed_size_obj = RandomBytesIO(cols=10, size=7777)
        self.fixed_size_file = os.path.join(self.dtemp, "fixed_size.csv")
        fixed_size_obj.save(self.fixed_size_file)

    def tearDown(self):
        shutil.rmtree(self.dtemp)

    def test_size(self):
        fixed_rows_obj = RandomBytesIO(cols=10, rows=50)
        self.assertEqual(os.stat(self.fixed_rows_file).st_size, fixed_rows_obj.size())
        fixed_size_obj = RandomBytesIO(cols=10, size=7777)
        self.assertEqual(os.stat(self.fixed_size_file).st_size, fixed_size_obj.size())

    def test_read(self):
        fixed_rows_obj = RandomBytesIO(cols=10, rows=50)
        with open(self.fixed_rows_file) as fd:
            for n in [7, 77, 777, 7007]:
                self.assertEqual(fixed_rows_obj.read(n), fd.read(n).encode("utf-8"))
                self.assertEqual(fixed_rows_obj.tell(), fd.tell())
        fixed_size_obj = RandomBytesIO(cols=10, size=7777)
        with open(self.fixed_size_file) as fd:
            for n in [7, 77, 777, 7007]:
                self.assertEqual(fixed_size_obj.read(n), fd.read(n).encode("utf-8"))
                self.assertEqual(fixed_size_obj.tell(), fd.tell())

    def test_read_all(self):
        fixed_rows_obj = RandomBytesIO(cols=10, rows=50)
        with open(self.fixed_rows_file) as fd:
            self.assertEqual(fixed_rows_obj.read(), fd.read().encode("utf-8"))
            self.assertEqual(fixed_rows_obj.tell(), fd.tell())
        fixed_size_obj = RandomBytesIO(cols=10, size=7777)
        with open(self.fixed_size_file) as fd:
            self.assertEqual(fixed_size_obj.read(), fd.read().encode("utf-8"))
            self.assertEqual(fixed_size_obj.tell(), fd.tell())

    def test_iter(self):
        fixed_rows_obj = RandomBytesIO(cols=10, rows=50)
        with open(self.fixed_rows_file) as fd:
            for row in fixed_rows_obj:
                self.assertEqual(row, fd.readline())
        fixed_size_obj = RandomBytesIO(cols=10, size=7777)
        with open(self.fixed_size_file) as fd:
            for row in fixed_size_obj:
                self.assertEqual(row, fd.readline())

    def test_readline(self):
        fixed_rows_obj = RandomBytesIO(cols=10, rows=50)
        with open(self.fixed_rows_file) as fd:
            for _ in range(50):
                self.assertEqual(fixed_rows_obj.readline(), fd.readline())
        fixed_size_obj = RandomBytesIO(cols=10, size=7777)
        with open(self.fixed_size_file) as fd:
            for _ in range(50):
                self.assertEqual(fixed_size_obj.readline(), fd.readline())

    def test_readlines(self):
        fixed_rows_obj = RandomBytesIO(cols=10, rows=50)
        with open(self.fixed_rows_file) as fd:
            self.assertEqual(fixed_rows_obj.readlines(), fd.readlines())
        fixed_size_obj = RandomBytesIO(cols=10, size=7777)
        with open(self.fixed_size_file) as fd:
            self.assertEqual(fixed_size_obj.readlines(), fd.readlines())


if __name__ == "__main__":
    ProgressiveTest.main()
