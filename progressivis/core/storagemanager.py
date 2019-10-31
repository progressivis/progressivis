import os
import tempfile
import unicodedata
import re
import shutil
import urllib
import logging
logger = logging.getLogger(__name__)

urljoin = urllib.parse.urljoin
pathname2url = urllib.request.pathname2url


class StorageManager(object):
    default = None

    def __init__(self, directory=None):
        self._directory = directory
        self.moduledir = dict()

    def start(self):
        if self._directory is None:
            self._directory = tempfile.mkdtemp(prefix='progressivis_')
            logger.debug('StorageManager creating directory %s',
                         self._directory)
        return self._directory

    def directory(self):
        return self.start()

    def module_directory(self, module):
        mid = module.name
        if mid in self.moduledir:
            return self.moduledir[mid]
        dirname = os.path.join(self.start(), mid)
        try:
            os.mkdir(dirname)
        except os.error:
            mid = sluggify(mid)
            dirname = os.path.join(self._directory, mid)
            os.mkdir(dirname)
        logger.debug('StorageManager creating module directory %s for %s',
                     dirname, id)
        self.moduledir[mid] = dirname
        return dirname

    def filename(self, name):
        return os.path.join(self.start(), name)

    def fullname(self, module, filename):
        return os.path.join(self.module_directory(module), filename)

    def url(self, module, filename):
        return urljoin('file:', pathname2url(self.fullname(module, filename)))

    def end(self):
        if self._directory is None:
            return False
        logger.debug('StorageManager removing directory %s', self._directory)
        shutil.rmtree(self._directory, ignore_errors=True)
        self._directory = None
        self.moduledir = {}


StorageManager.default = StorageManager()


def sluggify(s):
    slug = unicodedata.normalize('NFKD', s)
    slug = slug.encode('ascii', 'ignore').lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    slug = re.sub(r'[-]+', '-', slug)
    return slug
