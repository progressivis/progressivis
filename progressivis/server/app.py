import logging
log = logging.getLogger(__name__)

import flask

from progressivis.core.scheduler import Scheduler

class ProgressivisBlueprint(flask.Blueprint):
    def __init__(self, *args, **kwargs):
        super(ProgressivisBlueprint, self).__init__(*args, **kwargs)

    def setup(self, scheduler):
        self.scheduler = scheduler

#    
progressivis_bp = ProgressivisBlueprint('progressivis',
                                        'progressivis.server',
                                        static_folder='static',
                                        static_url_path='/progressivis/static',
                                        template_folder='templates')

def create_app(config="settings.py", scheduler=Scheduler.default):
    app = flask.Flask('progressivis.server')
    if isinstance(config, str):
        app.config.from_pyfile(config)
    elif isinstance(config, dict):
        app.config.update(config)

    app.register_blueprint(progressivis_bp)
    progressivis_bp.setup(scheduler)

    return app
