import sys
from functools import wraps
import os
import os.path
import tempfile
from sqlalchemy import (Table, Column, Integer, Float, String, Sequence, BLOB,
                        MetaData, ForeignKey, create_engine, select,
                        UniqueConstraint)
import uuid
import json
import numpy.distutils.cpuinfo as cpuinfo
import pandas as pd
from .utils import RunStepBenchEnv

    
def bench_this(to_decorate, module, env):
    """
    bench decorator
    """
    def bench_decorator(to_decorate):
        """
        This is the actual decorator. It brings togerther the function to be
        decorated and the decoration stuff
        """
        @wraps(to_decorate)
        def bench_wrapper(*args, **kwargs):
            """
            This function is the decoration
            run_step(self, run_number, step_size, howlong)
            """
            #import pdb;pdb.set_trace()
            run_number = args[0] if len(args)>0 else kwargs['run_number']
            step_size = args[1] if len(args)>1 else kwargs['step_size']
            howlong = args[2] if len(args)>2 else kwargs['howlong']
            ut, st, cut, cst, et = os.times()
            ret = to_decorate(*args, **kwargs) 
            ut_, st_, cut_, cst_, et_ = os.times()
            elapsed_time = et_ - et
            sys_time = st_ -st
            user_time = ut_ - ut
            ld_avg_1, ld_avg_5, ld_avg_15 = os.getloadavg()
            tbl = env.bench_table
            ins = tbl.insert().values(module_id=module.name,
                                          elapsed_time=elapsed_time,
                                          sys_time=sys_time,
                                          user_time=user_time,
                                          run_number=run_number,
                                          step_size=step_size,
                                          howlong=howlong,
                                          next_state=ret.get('next_state', 0),
                                          steps_run=ret.get('steps_run', 0),
                                          reads=ret.get('reads', 0),
                                          updates=ret.get('updates', 0),
                                          creates=ret.get('creates', 0)



                                          )
            with env.engine.connect() as conn:
                conn.execute(ins)           
            print(elapsed_time, sys_time, user_time)
            return ret
        return bench_wrapper
    return bench_decorator(to_decorate)

def decorate_module(m, env):
    assert hasattr(m, 'run_step')
    m.run_step = bench_this(to_decorate=m.run_step, module=m, env=env)
def decorate(scheduler, db_name, drop_existing_db=False):
    #import pdb;pdb.set_trace()
    env = RunStepBenchEnv(db_name=db_name, drop_existing_db=drop_existing_db)
    for m in scheduler.modules().values():
        decorate_module(m, env)
    return env
