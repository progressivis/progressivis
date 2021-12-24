from . import ProgressiveTest
import tempfile
import os

import fsspec  # type: ignore
import pandas as pd

from progressivis.io.csv_sniffer import CSVSniffer
from progressivis.datasets import get_dataset


def save_csv(suffix, contents):
    fd, tmpfile = tempfile.mkstemp(prefix="csv_sniffer", suffix=suffix)
    os.close(fd)
    with fsspec.open(tmpfile, mode="wb", compression="infer") as out:
        out.write(contents)
    return tmpfile


DF = pd.DataFrame(data={"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]})


def csv_with_delimiter(delimiter=","):
    return bytes("a,b,c\r\n1,2,3\r\n4,5,6\r\n7,8,9".replace(",", delimiter), "utf-8")


class TestCSVSniffer(ProgressiveTest):
    def test_01_simple(self):
        sniffer = CSVSniffer(get_dataset("bigfile"))
        dialect = sniffer.dialect()
        self.assertEqual(dialect.delimiter, ",")
        self.assertEqual(dialect.lineterminator, "\r\n")

    def with_delimiter(self, delimiter):
        tmpfile = save_csv(".csv", csv_with_delimiter(delimiter))
        try:
            sniffer = CSVSniffer(tmpfile)
            dialect = sniffer.dialect()
            self.assertEqual(dialect.delimiter, delimiter)
            self.assertEqual(dialect.lineterminator, "\r\n")
            df = sniffer.dataframe()
            self.assertTrue(DF.equals(df))
            # self.assertEqual(sniffer.params['names'], ['a', 'b', 'c'])
        finally:
            os.unlink(tmpfile)

    def test_03_delimiter(self):
        self.with_delimiter(";")
        self.with_delimiter("|")
        self.with_delimiter(":")
        self.with_delimiter("/")
        self.with_delimiter("&")
        self.with_delimiter("!")

    def test_04_usecols(self):
        tmpfile = save_csv(".csv", csv_with_delimiter())
        try:
            sniffer = CSVSniffer(tmpfile, usecols=["a", "b"])
            # dialect = sniffer.dialect()
            sniffer.dataframe()
            # self.assertEqual(sniffer.params['names'], ['a', 'b'])
            self.assertEqual(sniffer.params["usecols"], ["a", "b"])
        finally:
            os.unlink(tmpfile)
