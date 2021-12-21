from . import ProgressiveTest, skip


from progressivis.datasets import (
    get_dataset,
    get_dataset_bz2,
    get_dataset_gz,
    get_dataset_lzma,
)


class TestLoadDatasets(ProgressiveTest):
    def test_load_smallfile(self):
        _ = get_dataset("smallfile")

    def test_load_bigfile(self):
        _ = get_dataset("bigfile")

    def test_load_smallfile_bz2(self):
        _ = get_dataset_bz2("smallfile")

    def test_load_bigfile_bz2(self):
        _ = get_dataset_bz2("bigfile")

    def test_load_smallfile_gz(self):
        _ = get_dataset_gz("smallfile")

    def test_load_bigfile_gz(self):
        _ = get_dataset_gz("bigfile")

    def test_load_smallfile_lzma(self):
        _ = get_dataset_lzma("smallfile")

    @skip("Too slow ...")
    def test_load_bigfile_lzma(self):
        _ = get_dataset_lzma("bigfile")
