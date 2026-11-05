"Load a file from a remote site"
import logging
import os, os.path

from urllib import request, error
import ssl

logger = logging.getLogger(__name__)

## Disable certificate verification on CI
ctx = ssl.create_default_context()
if os.getenv("CI"):
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE


def wget_file(filename: str, url: str) -> str:
    if os.path.exists(filename):
        return filename
    attempts = 0
    while True:
        try:
            response = request.urlopen(url, timeout=5, context=ctx)
            content = response.read()
            with open(filename, "wb") as file:
                file.write(content)
            break
        except error.URLError as exc:
            attempts += 1
            logger.error("Failed to load %s", exc)
            if attempts == 3:
                raise
        logger.debug("Timeout, attempt: %d", attempts)
    return filename
