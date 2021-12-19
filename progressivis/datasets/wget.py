"Load a file from a remote site"
import logging
import os.path

from six.moves import urllib

logger = logging.getLogger(__name__)


def wget_file(filename, url):
    if os.path.exists(filename):
        return filename
    attempts = 0
    while True:
        try:
            response = urllib.request.urlopen(url, timeout=5)
            content = response.read()
            with open(filename, "wb") as file:
                file.write(content)
            break
        except urllib.error.URLError as exc:
            attempts += 1
            logger.error("Failed to load %s", exc)
            if attempts == 3:
                raise
        logger.debug("Timeout, attempt: %d", attempts)
    return filename
