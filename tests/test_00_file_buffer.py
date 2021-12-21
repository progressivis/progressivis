from progressivis.core.utils import filepath_to_buffer
from . import ProgressiveTest
import requests
import tempfile
import os

HTTP_URL = (
    "http://s3.amazonaws.com/h2o-release/h2o/master"
    "/1193/docs-website/resources/publicdata.html"
)
S3_URL = "s3://h2o-release/h2o/master/1193/docs-website" "/resources/publicdata.html"


class TestFileBuffer(ProgressiveTest):
    def setUp(self):
        req = requests.get(HTTP_URL)
        _, self.tmp_file = tempfile.mkstemp(prefix="p10s_", suffix=".html")
        with open(self.tmp_file, "wb") as f:
            f.write(req.content)

    def tearDown(self):
        os.remove(self.tmp_file)

    def test_file_buffer(self):
        reader_http, _, _, size_http = filepath_to_buffer(HTTP_URL)
        self.assertGreater(size_http, 0)
        reader_s3, _, _, size_s3 = filepath_to_buffer(S3_URL)
        self.assertEqual(size_http, size_s3)
        reader_file, _, _, size_file = filepath_to_buffer(self.tmp_file)
        self.assertEqual(size_file, size_s3)
        n1 = 12
        n2 = 34
        n3 = 56
        _ = reader_http.read(n1)
        _ = reader_http.read(n2)
        _ = reader_http.read(n3)
        self.assertEqual(reader_http.tell(), n1 + n2 + n3)
        _ = reader_s3.read(n1)
        _ = reader_s3.read(n2)
        _ = reader_s3.read(n3)
        self.assertEqual(reader_s3.tell(), n1 + n2 + n3)
        _ = reader_file.read(n1)
        _ = reader_file.read(n2)
        _ = reader_file.read(n3)
        self.assertEqual(reader_file.tell(), n1 + n2 + n3)
        try:
            reader_s3.close()
        except Exception:
            pass
        try:
            reader_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    ProgressiveTest.main()
