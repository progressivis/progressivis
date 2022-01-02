from __future__ import annotations


import os
import tempfile
import unicodedata
import re
import shutil
from urllib.parse import urljoin
from urllib.request import pathname2url
import logging

from typing import Optional, Dict, TYPE_CHECKING

FilePath = str

if TYPE_CHECKING:
    from .module import Module


logger = logging.getLogger(__name__)


class StorageManager:
    default: StorageManager

    def __init__(self, directory: Optional[FilePath] = None):
        self._directory = directory
        self.moduledir: Dict[str, FilePath] = dict()

    def start(self) -> FilePath:
        if self._directory is None:
            self._directory = tempfile.mkdtemp(prefix="progressivis_")
            logger.debug("StorageManager creating directory %s", self._directory)
        return self._directory

    def directory(self) -> FilePath:
        return self.start()

    def module_directory(self, module: Module) -> FilePath:
        mid = module.name
        if mid in self.moduledir:
            return self.moduledir[mid]
        dirname = os.path.join(self.start(), mid)
        try:
            os.mkdir(dirname)
        except os.error:
            mid = sluggify(mid)
            dirname = os.path.join(self.directory(), mid)
            os.mkdir(dirname)
        logger.debug("StorageManager creating module directory %s for %s", dirname, id)
        self.moduledir[mid] = dirname
        return dirname

    def filename(self, name: str) -> FilePath:
        return os.path.join(self.start(), name)

    def fullname(self, module: Module, filename: str) -> FilePath:
        return os.path.join(self.module_directory(module), filename)

    def url(self, module: Module, filename: str) -> str:
        return urljoin("file:", pathname2url(self.fullname(module, filename)))

    def end(self) -> None:
        if self._directory is None:
            return
        logger.debug("StorageManager removing directory %s", self._directory)
        shutil.rmtree(self._directory, ignore_errors=True)
        self._directory = None
        self.moduledir = {}


StorageManager.default = StorageManager()


def sluggify(s: str) -> str:
    slug = unicodedata.normalize("NFKD", s)
    slug = (slug.encode("ascii", "ignore")).decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    slug = re.sub(r"[-]+", "-", slug)
    return slug
