from enum import Enum
from uuid import uuid4
from inspect import getargspec

import pandas as pd
import numpy as np
import random

from progressive.core.common import ProgressiveError
from progressive.core.scheduler import *
from progressive.core.slot import *
from progressive.core.tracer import *
from progressive.core.time_predictor import *

default_quantum = 1

def connect(output_module, output_name, input_module, input_name):
    return output_module.connect_output(output_name, input_module, input_name)

class Module(TracerProxy):
    EMPTY_COLUMN = pd.Series([],index=[],name='empty')
    EMPTY_TIMESTAMP = np.nan # pd.to_datetime(np.nan)
    UPDATE_COLUMN = '_update'
    TRACE_SLOT = '_trace'

    state_created = 0
    state_ready = 1
    state_running = 2
    state_blocked = 3
    state_terminated = 4
    module_kwds = {
        'id',
        'quantum',
        'scheduler',
        'tracer',
        'predictor',
        'input_descriptors',
        'output_descriptors'}

    def __init__(self,
                 id=None,
                 quantum=None,
                 scheduler=None,
                 tracer=None,
                 predictor=None,
                 input_descriptors=[],
                 output_descriptors=[],
                 **kwds):
        if id is None:
            id = uuid4()
        if quantum is None:
            quantum = default_quantum
        if scheduler is None:
            scheduler = default_scheduler
        if tracer is None:
            tracer = default_tracer()
        if predictor is None:
            predictor = default_predictor()

        TracerProxy.__init__(self, tracer)

        # always present
        output_descriptors = output_descriptors + [SlotDescriptor(Module.TRACE_SLOT, type=pd.DataFrame, required=False)]
        
        self._id = id
        self.quantum = quantum
        self._scheduler = scheduler
        if self._scheduler.exists(id):
            raise ProgressiveError('module already exists in scheduler, delete it first')
        self.predictor = predictor
        self._start_time = None
        self._end_time = None
        self._last_update = None
        self._state = Module.state_created
        self._had_error = False
        self.input_descriptors = input_descriptors
        self.output_descriptors = output_descriptors
        self._input_slots = self._validate_descriptors(self.input_descriptors)
        self._input_types = {d.name: d.type for d in self.input_descriptors}
        self._output_slots = self._validate_descriptors(self.output_descriptors)
        self._output_types = {d.name: d.type for d in self.output_descriptors}
        self.default_step_size = 1
        self._scheduler.add(self)

    @staticmethod
    def _filter_kwds(kwds, function_or_method):
        argspec = getargspec(function_or_method)
        keys = argspec.args[len(argspec.args)-len(argspec.defaults):]
        filtered_kwds = {k: kwds[k] for k in kwds.viewkeys()&keys}
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
                raise ProgressiveError('Duplicate slot name %s in slot descriptor', desc.name)
            slots[desc.name] = None
        return slots

    def timer(self):
        return self._scheduler.timer()

    def describe(self):
        print 'id: %s' % self.id()
        print 'quantum: %f' % self.quantum
        print 'start_time: %s' % self._start_time
        print 'end_time: %s' % self._end_time
        print 'last_update: %s' % self._last_update
        print 'state: %s' % self._state
        print 'input_slots: %s' % self._input_slots
        print 'outpus_slots: %s' % self._output_slots
        print 'default_step_size: %f' % self.default_step_size

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return u'Module %s: %s' % (self.__class__.__name__, self.id())

    def __repr__(self):
        return self.__unicode__()

    def __hash__(self):
        return self._id.__hash__()

    def id(self):
        return self._id

    def scheduler(self):
        return self._scheduler

    def create_slot(self, output_name, input_module, input_name):
        return Slot(self, output_name, input_module, input_name)

    def connect_output(self, output_name, input_module, input_name):
        slot=self.create_slot(output_name, input_module, input_name)
        slot.connect()
        return slot

    def get_input_slot(self,name):
         # raise error is the slot is not declared
        return self._input_slots[name]

    def get_input_module(self,name):
        return self.get_input_slot(name).output_module

    def get_input_type(self,name):
        return self._input_types[name]

    def input_slot_values(self):
        return self._input_slots.values()

    def input_slot_names(self):
        return self._input_slots.keys()

    def _connect_input(self, slot):
        ret = self.get_input_slot(slot.input_name)
        self._input_slots[slot.input_name] = slot
        return ret

    def _disconnect_input(self, slot):
        pass

    def validate_inputs(self):
        # Only validate existence, the output code will test types
        for sd in self.input_descriptors:
            slot = self._input_slots[sd.name]
            if sd.required and slot is None:
                raise ProgressiveError('Missing inputs slot %s in %s',
                                       sd.name, self._id)

    def get_output_slot(self,name):
         # raise error is the slot is not declared
        return self._output_slots[name]

    def output_slot_type(self,name):
        return self._output_types[name]

    def output_slot_values(self):
        return self._output_slots.values()

    def output_slot_names(self):
        return self._output_slots.keys()

    def validate_outputs(self):
        for sd in self.output_descriptors:
            slot = self._output_slots[sd.name]
            if sd.required and slot is None:
                raise ProgressiveError('Missing output slot %s in %s',
                                       sd.name, self._id)
            if slot:
                slot.validate_types()

    def _connect_output(self, slot):
        slot_list = self.get_output_slot(slot.output_name)
        if slot_list is None:
            self._output_slots[slot.output_name] = [ slot ]
        else:
            slot_list.append(slot)
        return slot_list

    def _disconnect_output(self, slot):
        pass

    def validate_inouts(self):
        self.validate_inputs()
        self.validate_outputs()

    def validate(self):
        self.validate_inouts()
        self.state=Module.state_ready

    def get_data(self, name):
        if name==Module.TRACE_SLOT:
            return self.tracer.df()
        return None

    def update_timestamps(self):
        return EMPTY_COLUMN

    def run_step(self,step_size,howlong):
        """Run one step of the module, with a duration up to the 'howlong' parameter.

        Returns a dictionary with at least 5 pieces of information: 1)
        the new state among (ready, blocked, terminated),2) a number
        of read items, 3) a number of updated items (written), 4) a
        number of created items, and 5) the effective number of steps run.
        """
        raise NotImplementedError('run_step not defined')

    def _return_run_step(self, next_state, steps_run, reads=0, updates=0, creates=0):
        assert next_state>=Module.state_ready and next_state<=Module.state_blocked
        if creates and updates==0:
            updates=creates
        elif creates > updates:
            raise ProgressiveError('More creates (%d) than updates (%d)', creates, updates)
        return {'next_state': next_state,
                'steps_run': steps_run,
                'reads': reads,
                'updates': updates,
                'creates': creates}

    def is_created(self):
        return self._state==Module.state_created

    def is_running(self):
        return self._state == Module.state_running

    def is_ready(self):
        if self.state == Module.state_terminated:
            return False
        # source modules can be generators that
        # cannot run out of input, unless they decide so.
        if len(self._input_slots)==0:
            return True

        if self.state == Module.state_ready:
            return True

        if self.state == Module.state_blocked:
#            import pdb
#            pdb.set_trace()
            zombie=True
            for slot in self.input_slot_values():
                inmod = slot.output_module
                ints = inmod.last_update()
                ts = self.last_update()
                if (ints is not None and ts is None) or \
                  (ints > ts):
                    return True
                if inmod.state!=Module.state_terminated:
                    zombie = False

            if zombie:
                self.state = Module.state_terminated
        return False

    def is_terminated(self):
        return self._state==Module.state_terminated

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self.set_state(s)

    def set_state(self, s):
        assert s>=Module.state_created and s<=Module.state_terminated
        self._state = s

    def predict_step_size(self, duration):
        self.predictor.fit(self.trace_stats())
        return self.predictor.predict(duration, self.default_step_size)

    def start(self):
        pass

    def _stop(self):
        self._end_time = self._start_time
        self._last_update = self._end_time
        self._start_time = None

    def end(self):
        pass

    def last_update(self):
        return self._last_update

    def collect_stats(self, ts):
        return {}

    def run(self):
        if self.is_running():
            raise ProgressiveError('Module already running')
        self.state = Module.state_running
        now=self.timer()
        self._start_time = now
        self._end_time = self._start_time + self.quantum
        self.start_run(now, step_size=self.default_step_size, quantum=self.quantum)
        next_state = None
        step_size = self.default_step_size
        max_time = self.quantum / 4.0
        
        while self._start_time <= self._end_time:
            remaining_time = self._end_time-self._start_time
            # choose a step size around 25% of the quantum to allow the predictor to adjust
            step_size = self.predict_step_size(np.min([max_time, remaining_time]))
            #print 'Step_size: %d' % step_size
            if step_size == 0:
                break
            run_step_ret = {'reads': 0, 'updates': 0, 'creates': 0}
            try:
                self.before_run_step(now, step_size=step_size, quantum=self.quantum)
                run_step_ret = self.run_step(step_size, remaining_time)
                next_state = run_step_ret['next_state']
            except StopIteration:
                next_state = Module.state_terminated
                run_step_ret['next_state'] = next_state
                break
            except Exception as e:
                print "Exception in %s" % self.id()
                now = self.timer()
                self.exception(now, step_size=step_size, quantum=self.quantum)
                self._start_time = now
                self._stop()
                next_state = Module.state_terminated
                run_step_ret['next_state'] = next_state
                self._had_error = True
                raise e
            finally:
                now = self.timer()
                self.after_run_step(now, step_size=step_size, quantum=self.quantum,
                                    **run_step_ret)
                self.state = next_state
            if self._start_time is None or self.state != Module.state_ready:
                self.run_stopped(now, step_size=step_size, quantum=self.quantum)
                break
            self._start_time = now
        if self.state==Module.state_terminated:
            self.terminated(now, step_size=step_size, quantum=self.quantum)
        self.end_run(now, step_size=step_size, quantum=self.quantum)
        self._stop()


class Print(Module):
    def __init__(self, columns=None, **kwds):
        self._add_slots(kwds,'input_descriptors', [SlotDescriptor('in')])
        super(Print, self).__init__(quantum=0, **kwds)
        self._columns = columns

    def set_state(self, s):
        if s==Module.state_terminated:
            self._state = Module.state_blocked
        else:
            self._state = s
            
    def run_step(self, step_size, howlong):
        df = self.get_input_slot('in').data()
        steps=len(df)
        print df
        #print df.info()
        return self._return_run_step(Module.state_blocked, steps_run=len(df), reads=steps)
