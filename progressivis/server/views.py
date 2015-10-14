from __future__ import absolute_import

import logging
logger = logging.getLogger(__name__)

from StringIO import StringIO
from flask import render_template, request, send_from_directory, jsonify, abort
from os.path import join, dirname, abspath
from .app import progressivis_bp

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

@progressivis_bp.route('/progressivis/scheduler/', methods=['POST'])
def scheduler():
    scheduler = progressivis_bp.scheduler
    scheduler.set_tick_proc(progressivis_bp.tick_scheduler) # setting it multiple times is ok
    return jsonify(scheduler.to_json())
        
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

@progressivis_bp.route('/progressivis/module/<id>', methods=['GET', 'POST'])
def module(id):
    print 'Requested module %s'%id
    scheduler = progressivis_bp.scheduler
    module = scheduler.module[id]
    module.set_end_run(progressivis_bp.tick_module) # setting it multiple time is ok
    if request.method == 'POST':
        return jsonify(module.to_json())
    if module.is_visualization():
        vis = module.get_visualization()
        return render_template(vis+'.html', title="%s %s"%(vis,id), id=id)
    return render_template('module.html', title="Module "+id, id=id)

@progressivis_bp.route('/progressivis/module/<id>/image', methods=['GET'])
def module_image(id):
    print 'Requested module %s'%id
    scheduler = progressivis_bp.scheduler
    module = scheduler.module[id]
    if module is None:
        abort(404)
    img = module.get_image()
    if image is None:
        abort(404)
    return serve_pil_image(img)

def serve_pil_image(pil_img):
    img_io = StringIO()
    pil_img.save(img_io, 'PNG', compress_level=1)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

