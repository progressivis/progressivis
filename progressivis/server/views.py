"""
HTTP client for ProgressiVis server.
"""
from __future__ import absolute_import, division, print_function

from os.path import join, dirname, abspath
import logging

from flask import render_template, request, send_from_directory, jsonify, abort, send_file

import six

from .app import progressivis_bp, path_to_module, stop_server

logger = logging.getLogger(__name__)


SERVER_DIR = dirname(dirname(abspath(__file__)))
JS_DIR = join(SERVER_DIR, 'server/static')


@progressivis_bp.route('/progressivis/ping')
def _ping():
    return "pong"

@progressivis_bp.route('/progressivis/static/<path:filename>')
def progressivis_file(filename):
    "Path of JS dir"
    return send_from_directory(JS_DIR, filename)

@progressivis_bp.route('/')
@progressivis_bp.route('/progressivis/')
@progressivis_bp.route('/progressivis/scheduler.html')
def index(*unused_all, **kwargs):
    "Main entry"
    # pylint: disable=unused-argument
    return render_template('scheduler.html',
                           title="ProgressiVis Modules")

@progressivis_bp.route('/favicon.ico')
@progressivis_bp.route('/progressivis/favicon.ico')
def favicon():
    "Favorite icon"
    return send_from_directory(JS_DIR, 'favicon.ico', mimetype='image/x-icon')

@progressivis_bp.route('/progressivis/about.html')
def about(*unused_all, **kwargs):
    "About"
    # pylint: disable=unused-argument
    return render_template('about.html')

@progressivis_bp.route('/progressivis/contact.html')
def contact(*unused_all, **kwargs):
    "Contact"
    # pylint: disable=unused-argument
    return render_template('contact.html')

@progressivis_bp.route('/progressivis/module-graph.html')
def _module_graph(*unused_all, **kwargs):
    # pylint: disable=unused-argument
    return render_template('module_graph.html')

@progressivis_bp.route('/progressivis/debug/', defaults={'package': 'progressivis'})
@progressivis_bp.route('/progressivis/debug/package/<package>')
def _debug(package):
    logging.getLogger(package).setLevel(logging.DEBUG)
    return "OK"

@progressivis_bp.route('/progressivis/log', methods=['GET'])
def _log():
    return progressivis_bp.get_log()

@progressivis_bp.route('/progressivis/scheduler', methods=['POST', 'GET'])
def _scheduler():
    short = request.values.get('short', 'False').lower() != 'false'
    print('Scheduler short=', short, 'method=', request.method)
    sched = progressivis_bp.scheduler
    return jsonify(sched.to_json(short))

@progressivis_bp.route('/progressivis/scheduler/start', methods=['POST'])
def _scheduler_start():
    scheduler = progressivis_bp.scheduler
    if scheduler.is_running():
        return jsonify({'status': 'failed', 'reason': 'scheduler is already running'})
    #scheduler.start()
    progressivis_bp.start()
    return jsonify({'status': 'success'})

@progressivis_bp.route('/progressivis/scheduler/stop', methods=['POST'])
def _scheduler_stop():
    scheduler = progressivis_bp.scheduler
    if not scheduler.is_running():
        return jsonify({'status': 'failed', 'reason': 'scheduler is not is_running'})
    scheduler.stop()
    return jsonify({'status': 'success'})

@progressivis_bp.route('/progressivis/scheduler/step', methods=['POST'])
def _scheduler_step():
    scheduler = progressivis_bp.scheduler
    if scheduler.is_running():
        return jsonify({'status': 'failed', 'reason': 'scheduler is is_running'})
    #scheduler.step()
    progressivis_bp.step_once()
    return jsonify({'status': 'success'})

@progressivis_bp.route('/progressivis/module/get/<mid>', methods=['POST', 'GET'])
def _module(mid):
    module = path_to_module(mid)
    if module is None:
        abort(404)
    module.set_end_run(progressivis_bp.tick_module) # setting it multiple time is ok
    if request.method == 'POST':
        return jsonify(module.to_json())
    print('GET module %s'%mid)
    if module.is_visualization():
        vis = module.get_visualization()
        return render_template(vis+'.html', title="%s %s"%(vis, mid), id=mid)
    return render_template('module.html', title="Module "+mid, id=mid)

@progressivis_bp.route('/progressivis/module/image/<mid>', methods=['GET'])
def _module_image(mid):
    run_number = request.values.get('run_number', None)
    try:
        run_number = int(run_number)
    except ValueError:
        run_number = None
    print('Requested module image for %s?run_number=%s'%(mid, run_number))
    module = path_to_module(mid)
    if module is None:
        abort(404)
    img = module.get_image(run_number)
    if img is None:
        abort(404)
    if isinstance(img, six.string_types):
        return send_file(img, cache_timeout=0)
    return _serve_pil_image(img)

def _serve_pil_image(pil_img):
    img_io = six.StringIO()
    pil_img.save(img_io, 'PNG', compress_level=1)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', cache_timeout=0)

@progressivis_bp.route('/progressivis/module/set/<mid>', methods=['POST'])
def _module_set_parameter(mid):
    module = path_to_module(mid)
    if module is None:
        abort(404)
    var_values = request.get_json()
    try:
        module.set_current_params(var_values)
        # pylint: disable=broad-except
    except Exception as execpt:
        return jsonify({'status': 'failed', 'reason': 'Cannot set parameters: %s' % execpt})

    return jsonify({'status': 'success'})

@progressivis_bp.route('/progressivis/module/input/<path:path>', methods=['POST'])
def _module_input(path):
    module = path_to_module(path)
    if module is None:
        abort(405)
    var_values = request.get_json()
    msg = ''
    try:
        print('sending to %s: %s'%(module.id, var_values))
        msg = module.from_input(var_values)
        # pylint: disable=broad-except
    except Exception as exc:
        msg = str(exc)
        print('Error: %s'%msg)
        return jsonify({'status': 'failed', 'reason': 'Cannot input: %s' % msg})

    print('success: %s'%msg)
    ret = {'status': 'success'}
    if msg:
        ret['error'] = msg
    return jsonify(ret)

@progressivis_bp.route('/progressivis/module/df/<mid>/<slot>', methods=['GET', 'POST'])
def _df(mid, slot):
    module = path_to_module(mid)
    if module is None:
        abort(404)
    print('Getting slot "'+slot+'"')
    df = module.get_data(slot)
    if df is None:
        abort(404)
    if request.method == 'POST':
        return jsonify({'columns':['index']+df.columns})
    print('GET df %s/%s'%(mid, slot))
    return render_template('dataframe.html',
                           title="DataFrame "+mid+'/'+slot,
                           id=mid, slot=slot) #, df=df)

@progressivis_bp.route('/progressivis/module/quality/<mid>', methods=['POST'])
def _qual(mid):
    module = path_to_module(mid)
    if module is None:
        abort(404)
    slot = '_trace'
    print('Getting slot "'+slot+'"')
    df = module.get_data(slot)
    if df is None:
        abort(404)
    print('POST df %s/%s'%(mid, slot))
    qual = df['quality'].values
    return jsonify({'index':df.index.values, 'quality': qual})


@progressivis_bp.route('/progressivis/module/dfslice/<mid>/<slot>', methods=['POST'])
def _dfslice(mid, slot):
    module = path_to_module(mid)
    if module is None:
        abort(404)
    df = module.get_data(slot)
    if df is None:
        abort(404)
    start_ = int(request.form['start'])
    draw_ = int(request.form['draw'])
    length_ = int(request.form['length'])
    df_len = len(df)
    df_slice = df.iloc[start_:min(start_+length_, df_len)]
    print("reload slice", start_, 'len=', length_, 'table len=', df_len)
    return jsonify({'draw':draw_,
                    'recordsTotal':df_len,
                    'recordsFiltered':df_len,
                    'data': df_slice.to_json(orient='rows')})

@progressivis_bp.route('/exit')
def _exit_():
    stop_server()
    return "Stopped!"

@progressivis_bp.route('/progressivis/logger.html')
def _logger_page():
    managers = logging.Logger.manager.loggerDict
    ret = []
    for (module, log) in six.iteritems(managers):
        if isinstance(log, logging.Logger):
            ret.append({'module': module,
                        'level': logging.getLevelName(log.getEffectiveLevel())})
    def _key_log(a):
        return a['module'].lower()
    ret.sort(key=_key_log)
    return render_template('logger.html',
                           title="ProgressiVis Loggers", loggers=ret)

@progressivis_bp.route('/progressivis/logger', methods=['POST'])
def _logger():
    managers = logging.Logger.manager.loggerDict
    ret = []
    for (module, log) in six.iteritems(managers):
        if isinstance(log, logging.Logger):
            ret.append({'module': module,
                        'level': logging.getLevelName(log.getEffectiveLevel())})
    def _key_log(a):
        return a['module'].lower()
    ret.sort(key=_key_log)
    return jsonify({'loggers': ret})
