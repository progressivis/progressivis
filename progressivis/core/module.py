from __future__ import absolute_import, division, print_function

from traceback import print_exc
import re
import pdb
import pandas as pd
import numpy as np
import six
from abc import ABCMeta, abstractmethod
from progressivis.core.utils import (ProgressiveError, type_fullname)
from progressivis.table.table_base import BaseTable
from progressivis.table.table import Table
from progressivis.table.dshape import dshape_from_dtype
from progressivis.table.row import Row
from progressivis.core.slot import (SlotDescriptor, Slot,
                                    InputSlots, OutputSlots)
from progressivis.core.tracer_base import Tracer
from progressivis.core.time_predictor import TimePredictor
from progressivis.core.storagemanager import StorageManager
from progressivis.core.storage import Group
import logging
if six.PY2:  # pragma no cover
    from inspect import getargspec as getfullargspec
else:  # pragma no cover
    from inspect import getfullargspec

logger = logging.getLogger(__name__)


class ModuleMeta(ABCMeta):
    """Module metaclass is needed to collect the input parameter list
    in the field ``all_parameters''.
    """
    def __init__(cls, name, bases, attrs):
        if "parameters" not in attrs:
            cls.parameters = []
        all_props = list(cls.parameters)
        for base in bases:
            all_props += getattr(base, "all_parameters", [])
        cls.all_parameters = all_props
        super(ModuleMeta, cls).__init__(name, bases, attrs)


@six.python_2_unicode_compatible
class Module(six.with_metaclass(ModuleMeta, object)):
    parameters = [('quantum', np.dtype(float), 1.0),
                  ('debug', np.dtype(bool), False)]
    TRACE_SLOT = '_trace'
    PARAMETERS_SLOT = '_params'

    state_created = 0
    state_ready = 1
    state_running = 2
    state_blocked = 3
    state_zombie = 4
    state_terminated = 5
    state_invalid = 6
    state_name = ['created', 'ready', 'running', 'blocked',
                  'zombie', 'terminated', 'invalid']

    def __init__(self,
                 mid=None,
                 name=None,
                 group=None,
                 scheduler=None,
                 tracer=None,
                 predictor=None,
                 storage=None,
                 storagegroup=None,
                 input_descriptors=None,
                 output_descriptors=None,
                 **kwds):
        if scheduler is None:
            from .scheduler import Scheduler
            scheduler = Scheduler.default
        self._scheduler = scheduler
        if name is not None:
            if mid is None:
                mid = name
            else:
                raise ValueError('Cannot use name (%s) and mid (%s)'
                                 ' at the same time', name, mid)
        if mid is None:
            mid = self._scheduler.generate_id(self.pretty_typename())
        self._id = mid
        if predictor is None:
            predictor = TimePredictor.default()
        predictor.id = mid
        self.predictor = predictor
        if storage is None:
            storage = StorageManager.default
        self.storage = storage
        if storagegroup is None:
            storagegroup = Group.default()
        self.storagegroup = storagegroup
        if tracer is None:
            tracer = Tracer.default(mid, storagegroup)

        # always present
        input_descriptors = input_descriptors or []
        output_descriptors = output_descriptors or []
        output_descriptors += [SlotDescriptor(Module.TRACE_SLOT,
                                              type=BaseTable,
                                              required=False)]
        input_descriptors += [SlotDescriptor(Module.PARAMETERS_SLOT,
                                             type=BaseTable,
                                             required=False)]
        self.order = None
        self._group = group
        if self._scheduler.exists(mid):
            raise ProgressiveError('module already exists in scheduler,'
                                   ' delete it first')
        self.tracer = tracer
        self._start_time = None
        self._end_time = None
        self._last_update = 0
        self._state = Module.state_created
        self._had_error = False
        self._parse_parameters(kwds)
        self._input_slots = self._validate_descriptors(input_descriptors)
        self.input_descriptors = {d.name: d for d in input_descriptors}
        self._output_slots = self._validate_descriptors(output_descriptors)
        self.output_descriptors = {d.name: d for d in output_descriptors}
        self.default_step_size = 100
        self.input = InputSlots(self)
        self.output = OutputSlots(self)
        self.scheduler().add_module(self)
        # callbacks
        self._start_run = None
        self._end_run = None
        self._synchronized_lock = self.scheduler().create_lock()

    def create_dependent_modules(self, *params, **kwds):  # pragma no cover
        """Create modules that this module depends on.
        """
        pass

    def get_progress(self):
        """Return a tuple of numbers (current,total) where current is `current`
        progress value and `total` is the total number of values to process;
        these values can change during the computations.
        """
        if not self.has_any_input():
            return (0, 0)
        slots = self.input_slot_values()
        progresses = []
        for slot in slots:
            if slot is not None:
                progresses.append(slot.output_module.get_progress())
        if len(progresses) == 1:
            return progresses[0]
        elif len(progresses) == 0:
            return (0, 0)
        pos = 0
        size = 0
        for p in progresses:
            pos += p[0]
            size += p[1]
        return (pos, size)

    def get_quality(self):
        """Quality value, should increase.
        """
        return 0.0

    def destroy(self):
        self.scheduler().remove_module(self)
        # TODO remove connections with the input and output modules

    @staticmethod
    def _filter_kwds(kwds, function_or_method):
        argspec = getfullargspec(function_or_method)
        keys_ = argspec.args[len(argspec.args)-(0 if argspec.defaults is None
                                                else len(argspec.defaults)):]
        filtered_kwds = {k: kwds[k] for k in six.viewkeys(kwds) & keys_}
        return filtered_kwds

    @staticmethod
    def _add_slots(kwds, kwd, slots):
        if kwd in kwds:
            kwds[kwd] += slots
        else:
            kwds[kwd] = slots

    @staticmethod
    def _validate_descriptors(descriptor_list):
        slots = {}
        for desc in descriptor_list:
            if desc.name in slots:
                raise ProgressiveError('Duplicate slot name %s'
                                       ' in slot descriptor', desc.name)
            slots[desc.name] = None
        return slots

    @property
    def debug(self):
        return self.params.debug

    @debug.setter
    def debug(self, b):
        # TODO: should change the run_number of the params
        self.params.debug = bool(b)

    @property
    def parameter(self):
        return self._params

    @property
    def lock(self):
        return self._synchronized_lock

    def _parse_parameters(self, kwds):
        # pylint: disable=no-member
        self._params = _create_table(self.generate_table_name("params"),
                                     self.all_parameters)
        self.params = Row(self._params)
        for (name, _, _) in self.all_parameters:
            if name in kwds:
                self.params[name] = kwds.pop(name)

    def generate_table_name(self, name):
        return "s{}_{}_{}".format(self.scheduler().id, self.id, name)

    def timer(self):
        return self._scheduler.timer()

    def to_json(self, short=False):
        s = self.scheduler()
        json = {
            'is_running': s.is_running(),
            'is_terminated': s.is_terminated(),
            'run_number': s.run_number(),
            'id': self.id,
            'classname': self.pretty_typename(),
            'is_visualization': self.is_visualization(),
            'last_update': self._last_update,
            'state': self.state_name[self._state],
            'quality': self.get_quality(),
            'progress': list(self.get_progress())
        }
        if self.order is not None:
            json['order'] = self.order

        if short:
            return json

        with self.lock:
            json.update({
                'start_time': self._start_time,
                'end_time': self._end_time,
                'input_slots': {k: _slot_to_json(s) for (k, s) in
                                six.iteritems(self._input_slots)},
                'output_slots': {k: _slot_to_json(s) for (k, s) in
                                 six.iteritems(self._output_slots)},
                'default_step_size': self.default_step_size,
                'parameters': self.current_params().to_json()
            })
        return json

    def from_input(self, msg):
        if 'debug' in msg:
            self.debug = msg['debug']

    def is_input(self):
        return False

    def get_image(self, run_number=None):  # pragma no cover
        # pylint: disable=unused-argument
        """
        Return an image geenrated by this module.
        """
        return None

    def describe(self):
        print('id: %s' % self.id)
        print('class: %s' % type_fullname(self))
        print('quantum: %f' % self.params.quantum)
        print('start_time: %s' % self._start_time)
        print('end_time: %s' % self._end_time)
        print('last_update: %s' % self._last_update)
        print('state: %s(%d)' % (self.state_name[self._state], self._state))
        print('input_slots: %s' % self._input_slots)
        print('outpus_slots: %s' % self._output_slots)
        print('default_step_size: %d' % self.default_step_size)
        if len(self._params):
            print('parameters: ')
            print(self._params)

    def pretty_typename(self):
        name = self.__class__.__name__
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        s1 = re.sub('_module$', '', s1)
        return s1

    def __str__(self):
        return six.u('Module %s: %s' % (self.__class__.__name__, self.id))

    def __repr__(self):
        return str(self)

#    def __hash__(self):
#        return self._id.__hash__()

    @property
    def id(self):
        return self._id

    def name(self):
        return id

    @property
    def group(self):
        return self._group

    def scheduler(self):
        return self._scheduler

    def start(self):
        self.scheduler().start()

    def terminate(self):
        self.state = Module.state_zombie

    def create_slot(self, output_name, input_module, input_name):
        return Slot(self, output_name, input_module, input_name)

    def connect_output(self, output_name, input_module, input_name):
        slot = self.create_slot(output_name, input_module, input_name)
        slot.connect()
        return slot

    def has_any_input(self):
        return any(self._input_slots.values())

    def get_input_slot(self, name):
        # raises error is the slot is not declared
        return self._input_slots[name]

    def get_input_module(self, name):
        return self.get_input_slot(name).output_module

    def input_slot_values(self):
        return list(self._input_slots.values())

    def input_slot_type(self, name):
        return self.input_descriptors[name].type

    def input_slot_required(self, name):
        return self.input_descriptors[name].required

    def input_slot_names(self):
        return list(self._input_slots.keys())

    def _connect_input(self, slot):
        ret = self.get_input_slot(slot.input_name)
        self._input_slots[slot.input_name] = slot
        return ret

    def _disconnect_input(self, slot):  # pragma no cover
        pass

    def validate_inputs(self):
        # Only validate existence, the output code will test types
        valid = True
        for sd in self.input_descriptors.values():
            slot = self._input_slots[sd.name]
            if sd.required and slot is None:
                logger.error('Missing inputs slot %s in %s', sd.name, self._id)
                valid = False
        return valid

    def has_any_output(self):
        return any(self._output_slots.values())

    def get_output_slot(self, name):
        # raise error is the slot is not declared
        return self._output_slots[name]

    def output_slot_type(self, name):
        return self.output_descriptors[name].type

    def output_slot_values(self):
        return list(self._output_slots.values())

    def output_slot_names(self):
        return list(self._output_slots.keys())

    def validate_outputs(self):
        valid = True
        for sd in self.output_descriptors.values():
            slots = self._output_slots[sd.name]
            if sd.required and (slots is None or len(slots) == 0):
                logger.error('Missing required output slot %s in %s',
                             sd.name, self._id)
                valid = False
            if slots:
                for slot in slots:
                    if not slot.validate_types():
                        valid = False
        return valid

    def _connect_output(self, slot):
        slot_list = self.get_output_slot(slot.output_name)
        if slot_list is None:
            self._output_slots[slot.output_name] = [slot]
        else:
            slot_list.append(slot)
        return slot_list

    def _disconnect_output(self, slot):
        pass

    def validate_inouts(self):
        return self.validate_inputs() and self.validate_outputs()

    def validate(self):
        if self.validate_inouts():
            self.state = Module.state_blocked
            return True
        else:
            self.state = Module.state_invalid
            return False

    def get_data(self, name):
        if name == Module.TRACE_SLOT:
            return self.tracer.trace_stats()
        elif name == Module.PARAMETERS_SLOT:
            return self._params
        return None

    @abstractmethod
    def run_step(self, run_number, step_size, howlong):  # pragma no cover
        """Run one step of the module, with a duration up to the 'howlong' parameter.

        Returns a dictionary with at least 5 pieces of information: 1)
        the new state among (ready, blocked, zombie),2) a number
        of read items, 3) a number of updated items (written), 4) a
        number of created items, and 5) the effective number of steps run.
        """
        raise NotImplementedError('run_step not defined')

    @staticmethod
    def next_state(slot):
        if slot.has_buffered():
            return Module.state_ready
        return Module.state_blocked

    def _return_run_step(self, next_state, steps_run,
                         reads=0, updates=0, creates=0):
        assert (next_state >= Module.state_ready and
                next_state <= Module.state_zombie)
        if creates and updates == 0:
            updates = creates
        elif creates > updates:
            raise ProgressiveError('More creates (%d) than updates (%d)',
                                   creates, updates)
        return {'next_state': next_state,
                'steps_run': steps_run,
                'reads': reads,
                'updates': updates,
                'creates': creates}

    def is_visualization(self):
        return False

    def get_visualization(self):
        return None

    def is_created(self):
        return self._state == Module.state_created

    def is_running(self):
        return self._state == Module.state_running

    def is_ready(self):
        if self.state == Module.state_zombie:
            logger.info("%s Not ready because it turned from zombie"
                        " to terminated", self.id)
            self.state = Module.state_terminated
            return False
        if self.state == Module.state_terminated:
            logger.info("%s Not ready because it terminated", self.id)
            return False
        if self.state == Module.state_invalid:
            logger.info("%s Not ready because it is invalid", self.id)
            return False
        # source modules can be generators that
        # cannot run out of input, unless they decide so.
        if not self.has_any_input():
            return True

        # Module is either a source or has buffered data to process
        if self.state == Module.state_ready:
            return True

        # Module is waiting for some input, test if some is available
        # to let it run. If all the input modules are terminated,
        # the module is blocked, cannot run any more, so it is terminated
        # too.
        if self.state == Module.state_blocked:
            slots = self.input_slot_values()
            in_count = 0
            term_count = 0
            ready_count = 0
            for slot in slots:
                if slot is None:  # slot not required and not connected
                    continue
                in_count += 1
                in_module = slot.output_module
                in_ts = in_module.last_update()
                ts = slot.last_update()

                # logger.debug('for %s[%s](%d)->%s(%d)',
                #              slot.input_module.id, slot.input_name, in_ts,
                #              slot.output_name, ts)
                if slot.has_buffered() or in_ts > ts:
                    ready_count += 1
                elif (in_module.is_terminated() or
                      in_module.state == Module.state_invalid):
                    term_count += 1

            # if all the input slot modules are terminated or invalid
            if (not self.is_input() and in_count != 0 and
               term_count == in_count):
                logger.info('%s becomes zombie because all its input slots'
                            ' are terminated', self.id)
                self.state = Module.state_zombie
                return False
            # sources are always ready, and when 1 is ready, the module is.
            return in_count == 0 or ready_count != 0
        logger.error("%s Not ready because is in weird state %s",
                     self.id, self.state_name[self.state])
        return False

    def cleanup_run(self, run_number):
        """Perform operations such as switching state from zombie to terminated.

        Resources could also be released for terminated modules.
        """
        if self.is_zombie():  # terminate modules that died in the previous run
            self.state = Module.state_terminated
        return run_number  # keep pylint happy

    def is_zombie(self):
        return self._state == Module.state_zombie

    def is_terminated(self):
        return self._state == Module.state_terminated

    def is_valid(self):
        return self._state != Module.state_invalid

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self.set_state(s)

    def set_state(self, s):
        assert (s >= Module.state_created and
                s <= Module.state_invalid), "State %s invalid in module %s" % (
                    s, self.id)
        self._state = s

    def trace_stats(self, max_runs=None):
        return self.tracer.trace_stats(max_runs)

    def predict_step_size(self, duration):
        self.predictor.fit(self.trace_stats())
        return self.predictor.predict(duration, self.default_step_size)

    def starting(self):
        pass

    def _stop(self, run_number):
        self._end_time = self._start_time
        self._last_update = run_number
        self._start_time = None
        assert self.state != self.state_running
        self.end_run(run_number)

    def set_start_run(self, start_run):
        if start_run is None or callable(start_run):
            self._start_run = start_run
        else:
            raise ProgressiveError('value should be callable or None',
                                   start_run)

    def start_run(self, run_number):
        if self._start_run:
            self._start_run(self, run_number)

    def set_end_run(self, end_run):
        if end_run is None or callable(end_run):
            self._end_run = end_run
        else:
            raise ProgressiveError('value should be callable or None', end_run)

    def end_run(self, run_number):
        if self._end_run:
            self._end_run(self, run_number)

    def ending(self):
        pass

    def last_update(self):
        return self._last_update

    def last_time(self):
        return self._end_time

    def _update_params(self, run_number):
        pslot = self.get_input_slot(self.PARAMETERS_SLOT)
        if pslot is None or pslot.output_module is None:  # optional slot
            return
        df = pslot.data()
        if df is None:
            return
        raise NotImplementedError('Updating parameters not implemented yet')

    def current_params(self):
        return self._params.loc[self._params.index[-1]]

    def set_current_params(self, v):
        if not isinstance(v, pd.Series):
            v = pd.Series(v, dtype=object)  # raises error if not compatible
        with self.lock:
            current = self.current_params()
            v = current.combine_first(v)  # fill-in missing values
            self._params.loc[self._params.index[-1]+1] = v
        return v

    def run(self, run_number):
        if self.is_running():
            raise ProgressiveError('Module already running')
        next_state = self.state
        exception = None
        now = self.timer()
        quantum = self.scheduler().fix_quantum(self, self.params.quantum)
        tracer = self.tracer
        if quantum == 0:
            quantum = 0.1
            logger.error('Quantum is 0 in %s, setting it to a'
                         ' reasonable value', self.id)
        self.state = Module.state_running
        self._start_time = now
        self._end_time = self._start_time + quantum
        self._update_params(run_number)

        # TODO Forcing 4 steps, not sure, change when the predictor improves
        max_time = quantum / 4.0

        run_step_ret = {'reads': 0, 'updates': 0, 'creates': 0}
        self.start_run(run_number)
        tracer.start_run(now, run_number)
        while self._start_time < self._end_time:
            remaining_time = self._end_time-self._start_time
            if remaining_time <= 0:
                logger.info('Late by %d s in module %s',
                            remaining_time, self.pretty_typename())
                break  # no need to try to squeeze anything
            logger.debug('Time remaining: %f in module %s',
                         remaining_time, self.pretty_typename())
            step_size = self.predict_step_size(np.min([max_time,
                                                       remaining_time]))
            logger.debug('step_size=%d in module %s',
                         step_size, self.pretty_typename())
            if step_size == 0:
                logger.debug('step_size of 0 in module %s',
                             self.pretty_typename())
                break
            # pylint: disable=broad-except
            try:
                tracer.before_run_step(now, run_number)
                if self.debug:
                    pdb.set_trace()
                run_step_ret = self.run_step(run_number,
                                             step_size,
                                             remaining_time)
                next_state = run_step_ret['next_state']
                now = self.timer()
            except StopIteration:
                logger.info('In Module.run(): Received a StopIteration')
                next_state = Module.state_zombie
                run_step_ret['next_state'] = next_state
                now = self.timer()
                break
            except Exception as e:
                print_exc()
                next_state = Module.state_zombie
                run_step_ret['next_state'] = next_state
                now = self.timer()
                logger.debug("Exception in %s", self.id)
                tracer.exception(now, run_number)
                exception = e
                self._had_error = True
                self._start_time = now
                break
            finally:
                assert (run_step_ret is not None), "Error: %s run_step_ret"\
                  " not returning a dict" % self.pretty_typename()
                if self.debug:
                    run_step_ret['debug'] = True
                tracer.after_run_step(now, run_number, **run_step_ret)
                self.state = next_state
                logger.debug('Next step is %s in module %s',
                             self.state_name[next_state],
                             self.pretty_typename())

            if self._start_time is None or self.state != Module.state_ready:
                tracer.run_stopped(now, run_number)
                break
            self._start_time = now
        self.state = next_state
        if self.state == Module.state_zombie:
            logger.debug('Module %s zombie', self.pretty_typename())
            tracer.terminated(now, run_number)
        progress = self.get_progress()
        tracer.end_run(now, run_number,
                       progress_current=progress[0], progress_max=progress[1],
                       quality=self.get_quality())
        self._stop(run_number)
        if exception:
            raise RuntimeError("{} {}".format(type(exception), exception))


def print_len(x):
    if x is not None:
        print(len(x))


class Every(Module):
    def __init__(self, proc=print_len, constant_time=True, **kwds):
        self._add_slots(kwds, 'input_descriptors', [SlotDescriptor('df')])
        super(Every, self).__init__(**kwds)
        self._proc = proc
        self._constant_time = constant_time

    def predict_step_size(self, duration):
        if self._constant_time:
            return 1
        return super(Every, self).predict_step_size(duration)

    def run_step(self, run_number, step_size, howlong):
        slot = self.get_input_slot('df')
        df = slot.data()
        reads = 0
        if df is not None:
            with slot.lock:
                reads = len(df)
                with self.scheduler().stdout_parent():
                    self._proc(df)
        return self._return_run_step(Module.state_blocked, steps_run=1,
                                     reads=reads)


def prt(x):
    print(x)


class Print(Every):
    def __init__(self, **kwds):
        if 'proc' not in kwds:
            kwds['proc'] = prt
        super(Print, self).__init__(quantum=0.1, constant_time=True, **kwds)

def _slot_to_json(slot):
    if slot is None:
        return None
    if isinstance(slot, list):
        return [_slot_to_json(s) for s in slot]
    return slot.to_json()

def _create_table(tname, columns):
    ds = ""
    data = {}
    for (name, dtype, val) in columns:
        if len(ds):
            ds += ','
        ds += '%s: %s'%(name, dshape_from_dtype(dtype))
        data[name] = val
    ds = '{'+ds+'}'
    table = Table(tname, dshape=ds)
    table.add(data)
    return table
