import os
import tempfile
import unicodedata
import re
import shutil
from urllib import pathname2url
from urlparse import urljoin

import logging
logger = logging.getLogger(__name__)

def sluggify(s):
    slug = unicodedata.normalize('NFKD', s)
    slug = slug.encode('ascii', 'ignore').lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    slug = re.sub(r'[-]+', '-', slug)

class StorageManager(object):
    default = None

    def __init__(self, directory=None):
        self.directory = directory
        self.moduledir = dict()

    def start(self):
        if self.directory is None:
            self.directory = tempfile.mkdtemp(prefix='progressivis_')
            logger.debug('StorageManager creating directory %s' % self.directory)
        return self.directory
        
    def directory(self):
        return self.start()

    def module_directory(self, module):
        id = unicode(module.id)
        if id in self.moduledir:
            return self.moduledir[id]
        dirname = os.path.join(self.start(), id)
        try:
            os.mkdir(dirname)
        except:
            dirname=sluggify(id)
            dirname = os.path.join(self.directory, id)
            os.mkdir(dirname)
        logger.debug('StorageManager creating module directory %s for module %s' % (dirname, id))
        self.moduledir[id] = dirname
        return dirname

    def fullname(self, module, filename):
        return os.path.join(self.module_directory(module), filename)

    def url(self, module, filename):
        return urljoin('file:', urllib.pathname2url(self.fullname(module, filename)))

    def end(self):
        if self.directory is None:
            return False
        logger.debug('StorageManager removing directory %s' % self.directory)
        shutil.rmtree(self.directory, ignore_errors=True)
        self.directory = None
        self.moduledir = {}

StorageManager.default = StorageManager()
