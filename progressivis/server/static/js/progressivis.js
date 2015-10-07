var ProgressiVis = {};

ProgressiVis.get_scheduler = function(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler',
	dataType: 'json'
    })
	.done(success)
	.fail(error);
};

ProgressiVis.scheduler_start = function(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler/start',
	dataType: 'json'
    })
	.done(success)
	.fail(error);
};

ProgressiVis.scheduler_stop = function(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler/stop',
	dataType: 'json'
    })
	.done(success)
	.fail(error);
};
