"""
HTTP client for ProgressiVis server.
"""
from os.path import join
import logging
import io

import numpy as np

from aiohttp import web
import aiohttp_jinja2

from .app import progressivis_bp, path_to_module, stop_server, PROJECT_ROOT
from ..core import JSONEncoderNp
from ..utils.psdict import PsDict
from ..table.paging_helper import PagingHelper

routes = web.RouteTableDef()

logger = logging.getLogger(__name__)


JS_DIR = join(PROJECT_ROOT, 'static')


def _json_response(*args, **kwargs):
    return web.json_response(*args,
                             dumps=JSONEncoderNp.dumps,
                             **kwargs)


def _url_for(base_, filename=None):
    static_ = "/progressivis/static/"
    if base_ == 'progressivis.server.about':
        return f"{static_}about.html"
    if base_ == 'progressivis.server.contact':
        return f"{static_}contact.html"
    if base_ == 'progressivis.server.index':
        return "/progressivis/"
    return f"/progressivis/static/{filename}"


def render_template(tmpl, req, **kwargs):
    gen_ctx = {'url_for': _url_for, 'request': req, 'script_root': ''}
    new_ctx = {**gen_ctx, **kwargs}
    return aiohttp_jinja2.render_template(tmpl, req, new_ctx)


@routes.get('/ping')
async def _ping(_):
    return web.Response(text="pong")


@routes.get('/static/{filename}')
def progressivis_file(request):
    "Path of JS dir"
    filename = request.match_info['filename']
    return web.FileResponse(join(JS_DIR, filename))


@routes.get('/')
@routes.get('/scheduler.html')
async def index(request):
    "Main entry"
    # pylint: disable=unused-argument
    return render_template('scheduler.html', request,
                           title="ProgressiVis Modules")


@routes.get('/favicon.ico')
async def favicon(request):
    "Favorite icon"
    return web.FileResponse(join(JS_DIR, 'favicon.ico'))


@routes.get('/about.html')
async def about(request):
    "About"
    # pylint: disable=unused-argument
    return render_template('about.html', request)


@routes.get('/contact.html')
async def contact(request):
    "Contact"
    # pylint: disable=unused-argument
    return render_template('contact.html', request)


@routes.get('/module-graph.html')
async def _module_graph(request):
    # pylint: disable=unused-argument
    return render_template('module_graph.html', request)


# @routes.get('/debug/', defaults={'package': 'progressivis'})
@routes.get('/debug/package/{package}')
async def _debug(request):
    package = request.match_info['package']
    logging.getLogger(package).setLevel(logging.DEBUG)
    return "OK"


@routes.get('/log')
async def _log(_):
    return progressivis_bp.get_log()


@routes.get('/scheduler')
@routes.post('/scheduler')
async def _scheduler(request):
    short = request.values.get('short', 'False').lower() != 'false'
    print('Scheduler short=', short, 'method=', request.method)
    sched = progressivis_bp.scheduler
    return _json_response(sched.to_json(short))


@routes.post('/scheduler/start')
async def _scheduler_start(request):
    scheduler = progressivis_bp.scheduler
    if scheduler.is_running():
        return _json_response({'status': 'failed',
                               'reason': 'scheduler is already running'})
    progressivis_bp.start()
    return _json_response({'status': 'success'})


@routes.post('/scheduler/stop')
async def _scheduler_stop(request):
    scheduler = progressivis_bp.scheduler
    if not scheduler.is_running():
        return _json_response({'status': 'failed',
                               'reason': 'scheduler is not is_running'})
    scheduler.stop()
    return _json_response({'status': 'success'})


@routes.post('/scheduler/step')
async def _scheduler_step(request):
    scheduler = progressivis_bp.scheduler
    if scheduler.is_running():
        return _json_response({'status': 'failed',
                               'reason': 'scheduler is is_running'})
    progressivis_bp.step_once()
    return _json_response({'status': 'success'})


@routes.get('/module/get/{mid}')
@routes.post('/module/get/{mid}')
async def _module(request):
    mid = request.match_info['mid']
    module = path_to_module(mid)
    if module is None:
        raise web.HTTPNotFound()
    # setting it multiple time is ok
    module.set_end_run(progressivis_bp.tick_module)
    if request.method == 'POST':
        return _json_response(module.to_json())
    print('GET module %s' % mid)
    if module.is_visualization():
        vis = module.get_visualization()
        return render_template(vis+'.html',
                               request,
                               title="%s %s" % (vis, mid),
                               id=mid)
    return render_template('module.html', request, title="Module "+mid, id=mid)


@routes.get('/module/image/{mid}')
async def _module_image(request):
    mid = request.match_info['mid']
    run_number = request.values.get('run_number', None)
    try:
        run_number = int(run_number)
    except ValueError:
        run_number = None
    print('Requested module image for %s?run_number=%s' % (mid, run_number))
    module = path_to_module(mid)
    if module is None:
        raise web.HTTPNotFound()
    img = module.get_image(run_number)
    if img is None:
        raise web.HTTPNotFound()
    if isinstance(img, str):
        # send_file(img, cache_timeout=0)
        return web.FileResponse(img)
    return _serve_pil_image(img)


def _serve_pil_image(pil_img, *unused_all, **kwargs):
    img_io = io.StringIO()
    pil_img.save(img_io, 'PNG', compress_level=1)
    img_io.seek(0)
    # return send_file(img_io, mimetype='image/png', cache_timeout=0)
    return web.Response(body=img_io, content_type='image/png')


@routes.post('/module/set/{mid}')
async def _module_set_parameter(request):
    mid = request.match_info['mid']
    module = path_to_module(mid)
    if module is None:
        raise web.HTTPNotFound()
    var_values = request.get_json()
    try:
        module.set_current_params(var_values)
        # pylint: disable=broad-except
    except Exception as execpt:
        return _json_response({'status': 'failed',
                               'reason': 'Cannot set parameters: %s' % execpt})

    return _json_response({'status': 'success'})


@routes.post('/module/input/{path}')
async def _module_input(request):
    path = request.match_info['path']
    module = path_to_module(path)
    if module is None:
        raise web.HTTPNotFound()
    var_values = request.get_json()
    msg = ''
    try:
        print('sending to %s: %s' % (module.name, var_values))
        msg = module.from_input(var_values)
        # pylint: disable=broad-except
    except Exception as exc:
        msg = str(exc)
        print('Error: %s' % msg)
        return _json_response({'status': 'failed',
                               'reason': 'Cannot input: %s' % msg})
    print('success: %s' % msg)
    ret = {'status': 'success'}
    if msg:
        ret['error'] = msg
    return _json_response(ret)


@routes.get('/module/df/{mid}/{slot}')
@routes.post('/module/df/{mid}/{slot}')
async def _df(request):
    mid = request.match_info['mid']
    slot = request.match_info['slot']
    module = path_to_module(mid)
    if module is None:
        raise web.HTTPNotFound()
    print('Getting slot "'+slot+'"')
    df = module.get_data(slot)
    if df is None:
        raise web.HTTPNotFound()
    if request.method == 'POST':
        if isinstance(df, PsDict):
            return _json_response({'columns': ['index']+list(df.keys())})
        else:
            return _json_response({'columns': ['index']+df.columns})
    print('GET df %s/%s' % (mid, slot))
    return render_template('dataframe.html', request,
                           title="DataFrame "+mid+'/'+slot,
                           id=mid, slot=slot)  # df=df)


@routes.post('/module/quality/{mid}')
async def _qual(request):
    mid = request.match_info['mid']
    module = path_to_module(mid)
    if module is None:
        raise web.HTTPNotFound()
    slot = '_trace'
    print('Getting slot "'+slot+'"')
    df = module.get_data(slot)
    if df is None:
        raise web.HTTPNotFound()
    print('POST df %s/%s' % (mid, slot))
    qual = df['quality'].values[np.nonzero(df['quality'].values > 0)]
    return _json_response({'index': df.index.values, 'quality': qual})


@routes.post('/module/dfslice/{mid}/{slot}')
async def _dfslice(request):
    mid = request.match_info['mid']
    slot = request.match_info['slot']
    module = path_to_module(mid)
    if module is None:
        return web.HTTPNotFound()
    df = module.get_data(slot)
    if df is None:
        web.HTTPNotFound()
    form = await request.post()
    start_ = int(form['start'])
    draw_ = int(form['draw'])
    length_ = int(form['length'])
    # if slot in ('table', 'min_out', 'max_out'):
    #    import pdb; pdb.set_trace()
    if isinstance(df, PsDict):
        return _json_response({'draw': draw_,
                               'recordsTotal': 1,
                               'recordsFiltered': 1,
                               'data': [[0]+list(df.values())]})
    df_len = len(df)
    # df_slice = df.iloc[start_:min(start_+length_, df_len)]
    helper = PagingHelper(df)
    df_slice = helper.get_page(start_, min(start_+length_, df_len))
    print("reload slice", start_, 'len=', length_, 'table len=', df_len)
    return _json_response({'draw': draw_,
                           'recordsTotal': df_len,
                           'recordsFiltered': df_len,
                           'data': df_slice})


@routes.get('/exit')
async def _exit_(_):
    stop_server()
    return "Stopped!"


@routes.get('/logger.html')
async def _logger_page(request):
    managers = logging.Logger.manager.loggerDict
    ret = []
    for (module, log) in managers.items():
        if isinstance(log, logging.Logger):
            ret.append({'module': module,
                        'level': logging.getLevelName(log.getEffectiveLevel())
                        })

    def _key_log(a):
        return a['module'].lower()
    ret.sort(key=_key_log)
    return render_template('logger.html', request,
                           title="ProgressiVis Loggers", loggers=ret)


@routes.post('/logger')
async def _logger(_):
    managers = logging.Logger.manager.loggerDict
    ret = []
    for (module, log) in managers.items():
        if isinstance(log, logging.Logger):
            ret.append({'module': module,
                        'level': logging.getLevelName(log.getEffectiveLevel())
                        })

    def _key_log(a):
        return a['module'].lower()
    ret.sort(key=_key_log)
    return _json_response({'loggers': ret})


# import pdb;pdb.set_trace()
progressivis_bp.add_routes(routes)
