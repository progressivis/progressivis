from flask import Flask, request
from flask.globals import current_app
import time
import six
if six.PY3:
    from urllib.parse import urlparse as parse_url
    from urllib.parse import parse_qs
else:
    from urlparse import urlparse as parse_url
    from urlparse import parse_qs
try:
    from werkzeug.wsgi import wrap_file
except ImportError:
    from werkzeug.utils import wrap_file
from werkzeug.datastructures import Headers
from progressivis.core.utils import (RandomBytesIO,
                                         make_csv_fifo, del_tmp_csv_fifo)
app = Flask('stool.app')
@app.route('/buffer')
def get_buffer():
    dict_ = parse_qs(request.query_string)
    kwargs =  dict([(k, int(e[0])) for (k, e) in dict_.items()])
    rbio = RandomBytesIO(**kwargs)
    fsize = rbio.size()
    headers = Headers()
    headers['Content-Length'] = fsize
    filename = make_csv_fifo(rbio)
    file_ = open(filename, 'rb')
    data = wrap_file(request.environ, file_)
    return current_app.response_class(data, mimetype='text/csv', headers=headers,
                                    direct_passthrough=True)
    

if __name__ == '__main__':
    app.run()
