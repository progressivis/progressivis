from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)

from progressivis.core import Module

from StringIO import StringIO
from flask import render_template, request, send_from_directory, jsonify, abort, send_file
from os.path import join, dirname, abspath
from .app import progressivis_bp

#from pprint import pprint

SERVER_DIR = dirname(dirname(abspath(__file__)))
JS_DIR = join(SERVER_DIR, 'server/static')


@progressivis_bp.route('/progressivis/ping')
def ping():
    return "pong"

@progressivis_bp.route('/progressivis/static/<path:filename>')
def progressivis_file(filename):
    return send_from_directory(JS_DIR, filename)

@progressivis_bp.route('/')
@progressivis_bp.route('/progressivis/')
@progressivis_bp.route('/progressivis/scheduler.html')
def index(*unused_all, **kwargs):
    return render_template('scheduler.html',
                           title="ProgressiVis Modules")

@progressivis_bp.route('/favicon.ico')
@progressivis_bp.route('/progressivis/favicon.ico')
def favicon():
    return send_from_directory(JS_DIR, 'favicon.ico', mimetype='image/x-icon')

@progressivis_bp.route('/progressivis/about.html')
def about(*unused_all, **kwargs):
    return render_template('about.html')

@progressivis_bp.route('/progressivis/contact.html')
def contact(*unused_all, **kwargs):
    return render_template('contact.html')

@progressivis_bp.route('/progressivis/module-graph.html')
def module_graph(*unused_all, **kwargs):
    return render_template('module_graph.html')

@progressivis_bp.route('/progressivis/debug/', defaults={'package': 'progressivis', 'level': 'debug'})
@progressivis_bp.route('/progressivis/debug/package/<package>', defaults={'level': 'debug'})
def debug(package):
    logging.getLogger(package).setLevel(logging.DEBUG)
    return "OK"

@progressivis_bp.route('/progressivis/scheduler/', methods=['POST'])
def scheduler():
    short = request.values.get('short', 'True').lower() != 'false'
    scheduler = progressivis_bp.scheduler
    scheduler.set_tick_proc(progressivis_bp.tick_scheduler) # setting it multiple times is ok
    return jsonify(scheduler.to_json(short))

@progressivis_bp.route('/progressivis/scheduler/start', methods=['POST'])
def scheduler_start():
    scheduler = progressivis_bp.scheduler
    if scheduler.is_running():
        return jsonify({'status': 'failed', 'reason': 'scheduler is already running'})
    scheduler.start()
    return jsonify({'status': 'success'})

@progressivis_bp.route('/progressivis/scheduler/stop', methods=['POST'])
def scheduler_stop():
    scheduler = progressivis_bp.scheduler
    if not scheduler.is_running():
        return jsonify({'status': 'failed', 'reason': 'scheduler is not is_running'})
    scheduler.stop()
    return jsonify({'status': 'success'})

@progressivis_bp.route('/progressivis/scheduler/step', methods=['POST'])
def scheduler_step():
    scheduler = progressivis_bp.scheduler
    if scheduler.is_running():
        return jsonify({'status': 'failed', 'reason': 'scheduler is is_running'})
    scheduler.step()
    return jsonify({'status': 'success'})

def path_to_module(path):
    print 'module_path(%s)'%(path)
    ids = path.split('/')
    
    scheduler = progressivis_bp.scheduler
    module = scheduler.module[ids[0]]
    if module is None:
        return None
    for subid in ids[1:]:
        if not hasattr(module, subid):
            return None
        module = getattr(module, subid)
        if not isinstance(module, Module):
            return None
    return module

@progressivis_bp.route('/progressivis/module/get/<id>', methods=['GET', 'POST'])
def module(id):
    module = path_to_module(id)
    if module is None:
        abort(404)
    module.set_end_run(progressivis_bp.tick_module) # setting it multiple time is ok
    if request.method == 'POST':
        print 'POST module %s'%id
        return jsonify(module.to_json())
    print 'GET module %s'%id
    if module.is_visualization():
        vis = module.get_visualization()
        return render_template(vis+'.html', title="%s %s"%(vis,id), id=id)
    return render_template('module.html', title="Module "+id, id=id)

@progressivis_bp.route('/progressivis/module/image/<id>', methods=['GET'])
def module_image(id):
    run_number = request.values.get('run_number', None)
    try:
        run_number = int(run_number)
    except:
        run_number = None
    print 'Requested module image for %s?run_number=%s'%(id,run_number)
    module = path_to_module(id)
    if module is None:
        abort(404)
    img = module.get_image(run_number)
    if img is None:
        abort(404)
    if isinstance(img, (str, unicode)):
        return send_file(img, cache_timeout=0)
    return serve_pil_image(img)

def serve_pil_image(pil_img):
    img_io = StringIO()
    pil_img.save(img_io, 'PNG', compress_level=1)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', cache_timeout=0)

@progressivis_bp.route('/progressivis/module/set/<id>', methods=['POST'])
def module_set_parameter(id):
    module = path_to_module(id)
    if module is None:
        abort(404)
    var_values = request.get_json()
    try:
        module.set_current_params(var_values)
    except Exception as e:
        return jsonify({'status': 'failed', 'reason': 'Cannot set parameters: %s' % e})

    return jsonify({'status': 'success'})

@progressivis_bp.route('/progressivis/module/input/<path:path>', methods=['POST'])
def module_input(path):
    module = path_to_module(path)
    if module is None:
        abort(404)
    var_values = request.get_json()
    msg = ''
    try:
        print 'sending to %s: %s'%(module.id, var_values)
        msg = module.from_input(var_values)
    except Exception as e:
        msg = str(e)
        print 'Error: %s'%msg
        return jsonify({'status': 'failed', 'reason': 'Cannot input: %s' % msg})

    print 'success: %s'%msg
    ret = {'status': 'success'}
    if msg:
        ret['error'] = msg
    return jsonify(ret)

@progressivis_bp.route('/progressivis/module/df/<id>/<slot>', methods=['GET', 'POST'])
def df(id,slot):
    module = path_to_module(id)
    if module is None:
        abort(404)
    print 'Getting slot "'+slot+'"'
    df = module.get_data(slot)
    if df is None:
        abort(404)
    if request.method == 'POST':
        print 'POST df %s/%s'%(id,slot)
        return jsonify(df.to_dict(orient='split'))
    print 'GET df %s/%s'%(id,slot)
    return render_template('dataframe.html', title="DataFrame "+id+'/'+slot, id=id, slot=slot, df=df)

