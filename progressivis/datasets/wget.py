from six.moves import urllib
import logging
logger = logging.getLogger(__name__)

def wget_file(filename, url):
    attempts = 0
    while attempts < 3:
        try:
            response = urllib.request.urlopen(url, timeout = 5)
            content = response.read()
            with open(filename, 'wb' ) as f:
                f.write(content)
            break
        except urllib.error.URLError as e:
            attempts += 1
            logger.error('Failed to load %s',e)
    return filename
