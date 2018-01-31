from __future__ import absolute_import, division, print_function

from six import StringIO
import six
from os.path import join, dirname, abspath
import logging
logger = logging.getLogger(__name__)

from flask import render_template, request, send_from_directory, jsonify, abort, send_file
from .app import progressivis_bp
import tornado.ioloop

from progressivis.core import Module


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

@progressivis_bp.route('/progressivis/debug/', defaults={'package': 'progressivis'})
@progressivis_bp.route('/progressivis/debug/package/<package>')
def debug(package):
    logging.getLogger(package).setLevel(logging.DEBUG)
    return "OK"

@progressivis_bp.route('/progressivis/log', methods=['GET'])
def log():
    return progressivis_bp.get_log()

@progressivis_bp.route('/progressivis/scheduler/', methods=['POST'])
def scheduler():
    short = request.values.get('short', 'True').lower() != 'false'
    sched = progressivis_bp.scheduler
    #sched.set_tick_proc(progressivis_bp.tick_scheduler) # setting it multiple times is ok
    d = sched.to_json(short)
    return jsonify(d)

@progressivis_bp.route('/progressivis/scheduler/start', methods=['POST'])
def scheduler_start():
    scheduler = progressivis_bp.scheduler
    if scheduler.is_running():
        return jsonify({'status': 'failed', 'reason': 'scheduler is already running'})
    #scheduler.start()
    progressivis_bp.start()
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
    #scheduler.step()
    progressivis_bp.step_once()
    return jsonify({'status': 'success'})

def path_to_module(path):
    print('module_path(%s)'%(path))
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
        print('POST module %s'%id)
        d = module.to_json()
        return jsonify(d)
    print('GET module %s'%id)
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
    print('Requested module image for %s?run_number=%s'%(id,run_number))
    module = path_to_module(id)
    if module is None:
        abort(404)
    img = module.get_image(run_number)
    if img is None:
        abort(404)
    if isinstance(img, six.string_types):
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
        abort(405)
    var_values = request.get_json()
    msg = ''
    try:
        print('sending to %s: %s'%(module.id, var_values))
        msg = module.from_input(var_values)
    except Exception as e:
        msg = str(e)
        print('Error: %s'%msg)
        return jsonify({'status': 'failed', 'reason': 'Cannot input: %s' % msg})

    print('success: %s'%msg)
    ret = {'status': 'success'}
    if msg:
        ret['error'] = msg
    return jsonify(ret)

@progressivis_bp.route('/progressivis/module/df/<id>/<slot>', methods=['GET', 'POST'])
def df(id,slot):
    module = path_to_module(id)
    if module is None:
        abort(404)
    print('Getting slot "'+slot+'"')
    df = module.get_data(slot)
    if df is None:
        abort(404)
    if request.method == 'POST':
        print('POST df %s/%s'%(id,slot))
        #ret = jsonify(df.to_json(orient='split'))
        return jsonify({'columns':['index']+df.columns})        
    print('GET df %s/%s'%(id,slot))
    return render_template('dataframe.html', title="DataFrame "+id+'/'+slot, id=id, slot=slot) #, df=df)

@progressivis_bp.route('/progressivis/module/quality/<id>', methods=['POST'])
def qual(id):
    module = path_to_module(id)
    if module is None:
        abort(404)
    slot = '_trace'
    print('Getting slot "'+slot+'"')
    df = module.get_data(slot)
    if df is None:
        abort(404)
    print('POST df %s/%s'%(id,slot))
    qual = df['quality'].values
    return jsonify({'index':df.index.values, 'quality': qual})        


@progressivis_bp.route('/progressivis/module/dfslice/<id>/<slot>', methods=['POST'])
def dfslice(id,slot):
    module = path_to_module(id)
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
    print("reload slice", start_)
    return jsonify({'draw':draw_, 'recordsTotal':df_len, 'recordsFiltered':df_len, 'data': df_slice.to_json(orient='rows')})
@progressivis_bp.route('/exit')
def exit_():
    tornado.ioloop.IOLoop.instance().stop()
    return "Stopped!"
