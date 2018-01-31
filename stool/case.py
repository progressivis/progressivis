import numpy as np
import pandas as pd
import six
from functools import wraps, partial
from memory_profiler import memory_usage
import sys, os, time
from collections import OrderedDict, Iterable, namedtuple
from sqlalchemy import (Table, Column, Integer, Float, String, Sequence, BLOB,
                        MetaData, ForeignKey, create_engine, select,
                        UniqueConstraint, text)
#from multiprocessing import Process
import matplotlib
import matplotlib.pyplot as plt
import platform
import cProfile
import subprocess, time
from operator import itemgetter, attrgetter
import operator
import types
import copy

import unittest
from . utils import *

bench_cnt = 0
def bench(f=None, name=None, **kw):
    def bench_decorator_impl(f):
        global bench_cnt
        f.bench_info = dict(name=name, pos=bench_cnt, **kw)
        bench_cnt += 1
        @wraps(f)
        def wrapped(inst, *args, **kwargs):
            return f(inst, *args, **kwargs)
        return wrapped
    if f is None:
        return bench_decorator_impl
    return bench_decorator_impl(f)

class BenchMeta(type):
    def __init__(cls, name, bases, attrs):
        global bench_cnt
        bench_cnt = 0
        super(BenchMeta, cls).__init__(name, bases, attrs)
        bench_list = []
        for obj in attrs.values():
            if hasattr(obj, 'bench_info'):
                dict_ = getattr(obj, 'bench_info')
                dict_['func'] = obj
                if dict_['name'] is None:
                    dict_['name'] = obj.__name__
                bench_list.append(dict_)
                delattr(obj, 'bench_info')
        cls._bench_list = sorted(bench_list, key=itemgetter('pos'))
        #if cls._bench_list:
        #    cls.runTest = lambda self: cls.runBench(self)
        
        
class BenchmarkCase(six.with_metaclass(BenchMeta, unittest.TestCase)):
    """
    Each case should be placed in a method decorated by @bench(case=<unique_name>, corrected_by)
    When subclassing BenchmarkIt, you can set these attributes:
    * nb_step : number of steps : conditions (e.g. data size) should be different between two steps
    * nb_repeat  : number of repetitions for each step, conditions are identical
    * description : ...
    * database_name : temporary file if not defined
    * with_memory_profile = True|False
    * with_time_profile = True|False
    * with_code_profile = True|False
    """
    nb_step = 1
    nb_repeat = 1
    with_memory_prof = True
    with_time_prof = True
    with_code_prof = False

    def __init__(self,  methodName='runTest'):
#        if methodName != 'runBench':
#            raise ValueError('only runBench method is allowed here')
        #print("methodName: ".format(methodName))
        self._env = None
        self._name = get_random_name("B")
        self.nb_step = 1
        self.nb_repeat = 1
        self._current_step = 0
        self.description = "Undefined description"
        #self._input_proc = InputProc(repr_type=str, label="My label")
        self._step_info = None
        self._step_header = "Steps"
        self.with_time_prof = True
        self.with_memory_prof = True
        self.with_code_prof = False
        self._sql_id = None
        self._cases = []
        self._corrections = []
        super(BenchmarkCase, self).__init__(methodName)
    def set_step_info(self, info):
        self._step_info = info
    def set_step_header(self, header):
        self._step_header = header
    @property
    def step_info(self):
        return (self._step_info if self._step_info is not None
                    else str(self._current_step))
    @property
    def step_header(self):
        return self._step_header
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def setUpStep(self, step):
        pass
    def tearDownStep(self, step):
        pass
    @property
    def current_step(self):
        return self._current_step
    def dump(self):
        lst_table=['bench_tbl', 'case_tbl','measurement_tbl']
        for tbl in lst_table:
            dump_table(tbl, self.env._db_name)
    def save(self, db_name, name=None):
        dict_ = {}
        for case_name in self.case_names:
            dict_[case_name] = self.to_df(case_name, corrected=False,
                                              raw_header=True,
                                              all_columns=True)
        new_obj = copy.copy(self)
        new_obj._env = BenchEnv(db_name=db_name)
        if name is not None:
            new_obj._name = name
        new_obj._sql_id = None
        new_obj.init_db_entry()
        ins = new_obj.env.measurement_tbl.insert() 
        with new_obj.env.engine.connect() as conn:
            for case_name, case_df in dict_.items():
                headers = case_df.columns.values
                for _, row in case_df.iterrows():
                    ins_val = dict(zip(headers, row))
                    case_id = new_obj.get_case_id(case_name)
                    ins_val['case_id'] = case_id
                    del ins_val['id']
                    conn.execute(ins, ins_val)
        return new_obj
    @staticmethod
    def load(db_name, name):
        bm =  BenchmarkCase()
        bm._env = BenchEnv(db_name=db_name)
        bm._name = name
        props = bm.load_bench_entry()
        if not props:
            return None
        bm._sql_id = props['id']
        bm._step_header = props['step_header']
        return bm
    def init_db_entry(self):
        tbl = self.env._bench_tbl
        ins = tbl.insert().values(name=self._name, description=self.description,
                                  step_header=self._step_header,
                                  py_version=platform.python_version(),
                                  py_compiler=platform.python_compiler(),
                                  py_platform=platform.platform(),
                                  py_impl=platform.python_implementation(),
                                  cpu_info=get_cpu_info(),                      
                                  )
        with self.env.engine.connect() as conn:
            conn.execute(ins)
        cls = type(self)
        tbl = self.env._case_tbl
        with self.env.engine.connect() as conn:
            for dict_ in cls._bench_list:
                name = dict_['name']
                func = dict_['func']
                corrected_by = dict_.get('corrected_by', None)
                self._corrections.append((name, corrected_by))
                ins = tbl.insert().values(name=name, bench_id=self.sql_id)
                conn.execute(ins)
                case_id = self.get_case_id(name)
                runner = BenchRunner(func, case_id, corrected_by, self)
                self._cases.append(runner)
        self._update_corrections()
        return self
    @property
    def sql_id(self):
        if self._sql_id is not None:
            return self._sql_id
        tbl = self.env._bench_tbl
        s = (select([tbl]).with_only_columns([tbl.c.id, tbl.c.name]).
             where(tbl.c.name==self._name))
        #http://www.rmunn.com/sqlalchemy-tutorial/tutorial.html
        with self.env.engine.connect() as conn:
            row = conn.execute(s).fetchone()
            self._sql_id = row[0]
            return self._sql_id
    def load_bench_entry(self):
        tbl = self.env._bench_tbl
        s = (select([tbl]).
             where(tbl.c.name==self._name))
        with self.env.engine.connect() as conn:
            row = conn.execute(s).fetchone()
            if not row:
                return None
            return dict(row)
    def _update_corrections(self):
        for case, corr_by in self._corrections:
            if corr_by is None:
                continue
            case_id = self.get_case_id(case)
            corr_id = self.get_case_id(corr_by)            
            tbl = self.env._case_tbl
            stmt = (tbl.update().where(tbl.c.id==case_id).
                    values(corrected_by=corr_id))
            with self.env.engine.connect() as conn:
                conn.execute(stmt)

    def get_case_id(self, case):
        tbl = self.env._case_tbl
        s = (select([tbl]).with_only_columns([tbl.c.id]).
             where(tbl.c.name==case).
             where(tbl.c.bench_id==self.sql_id))
        with self.env.engine.connect() as conn:
            row = conn.execute(s).fetchone()
            return row[0]
    def get_case_corrector(self, case):
        if isinstance(case, six.string_types):
            case_id = self.get_case_id(case)
        else:
            case_id = case
        tbl = self.env._case_tbl
        s = (select([tbl]).with_only_columns([tbl.c.corrected_by]).
             where(tbl.c.id==case_id))
        with self.env.engine.connect() as conn:
            row = conn.execute(s).fetchone()
            return row[0]
    @property
    def correctors(self):
        tbl = self.env._case_tbl
        s = select([tbl]).with_only_columns([tbl.c.corrected_by])
        with self.env.engine.connect() as conn:
            rows = conn.execute(s).fetchall()
            return [e[0] for e in rows if e[0] is not None] 

    def runBench(self):
        self.init_db_entry()
        for elt in self._cases:
            self._current_step = 0
            for step in range(1, self.nb_step+1):
                try:
                    self._current_step += 1
                    self.setUpStep(step)
                except unittest.SkipTest as e:
                    self._addSkip(result, str(e))
                except KeyboardInterrupt:
                    raise
                except:
                    #unittest.result.addError(self, sys.exc_info())
                    raise
                else:
                    elt.run(step)
                try:
                    self.tearDownStep(step)
                except KeyboardInterrupt:
                    raise
                except:
                    #unittest.result.addError(self, sys.exc_info())
                    success = False
                
    @property
    def _col_dict(self):
        return {'case': 'Case', 'corrected_by':'Corrected by', 'i_th':'Measure',
                'mem_usage': 'Memory usage', 'elapsed_time': 'Elapsed time',
                'sys_time': 'System time', 'user_time': 'User time',
                'ld_avg_1':'Load avg.(-1s)',
                'ld_avg_5':'Load avg.(-5s)','ld_avg_15':'Load avg.(-15s)',
                'step_info': self._step_header}
    @property
    def case_names(self):
        tbl = self.env._case_tbl
        s = (select([tbl]).with_only_columns([tbl.c.name]).
             where(tbl.c.bench_id==self.sql_id))
        with self.env.engine.connect() as conn:
            rows = conn.execute(s).fetchall()
            return set([elt.name for elt in rows])
        #df =  pd.read_sql_query(s, conn)
        #return set(df['name'].values)

    def step_col_df(self, case_id):
        projection = [text('step_info')]
        tbl = self.env.measurement_tbl
        s = (select([tbl]).with_only_columns(projection).
             where(tbl.c.case_id==case_id).
             order_by(tbl.c.id))
        conn = self.env.engine.connect()
        return pd.read_sql_query(s, conn)
    def df_subtract(self, this, other, sub_cols, raw_header):
        ret = []
        for col in this.columns:
            if col in sub_cols:
                arr = this[col].values - other[col].values
            else:
                arr = this[col].values
            key = col if raw_header else self._col_dict.get(col,col)
            ret.append((key, arr))
        return pd.DataFrame.from_items(ret)
    def pretty_header(self, df, raw_header):
        if raw_header:
            return df
        header = [self._col_dict.get(col,col) for col in df.columns]
        df.columns = header
        return df
    def to_df(self, case, with_mem=True, with_times=True,
              with_step_info=True, corrected=True, raw_header=False, all_columns=False):
        if isinstance(case, six.string_types):
            case_id = self.get_case_id(case)
        else:
            case_id = case
        projection = ['i_th']
        projection += ['step_info'] if with_step_info==True else []
        if  with_mem:
            projection.append('mem_usage')
        if with_times:
            projection += ['elapsed_time', 'sys_time', 'user_time']
        projection += ['ld_avg_1', 'ld_avg_5', 'ld_avg_15']
        tbl = self.env.measurement_tbl
        only_columns = [col for col in tbl.columns if col.name in projection]
        s = (select([tbl]).
             where(tbl.c.case_id==case_id).
             order_by(tbl.c.id)) #.with_only_columns(only_columns)
        if not all_columns:
            s = s.with_only_columns(only_columns)
        conn = self.env.engine.connect()
        df =  pd.read_sql_query(s, conn)#, index_col=index_col)
        if not corrected:
            return self.pretty_header(df, raw_header)
        corr = self.get_case_corrector(case_id)
        if corr is None:
            return self.pretty_header(df, raw_header) 
        corr_df = self.to_df(corr, corrected=False, raw_header=True)
        return self.df_subtract(df, corr_df, ['mem_usage','elapsed_time',
                                              'sys_time', 'user_time'],
                                raw_header)

    def __getitem__(self, case):
        return self.to_df(case, raw_header=True, all_columns=True)
    def plot(self, cases=None, x=None, y=None, corrected=True, plot_opt='all'):
        if cases is None:
            cases = self.case_names
        elif isinstance(cases, six.string_types):
            cases = set([cases])
        else:
            cases = set(cases)
        if not cases.issubset(self.case_names):
            raise ValueError("Unknown case(s): {}".format(case))
        #df = self.to_df(raw_header=True)
        from matplotlib.lines import Line2D
        favorite_colors = ['red', 'blue', 'magenta', 'orange', 'grey',
                           'yellow', 'black']
        colors = list(matplotlib.colors.cnames.keys())
        customized_colors = (favorite_colors +
                             [c for c in colors if c not in favorite_colors])
        Bplot = namedtuple('Bplot','key, title, ylabel')
        mode = "corrected mode, show " if corrected else "raw mode, show "
        mode += plot_opt 
        plots = [Bplot('mem_usage', '[{}] Memory usage ({})'.format(self._name, mode),
                       'Used memory (Mb)'),
                 Bplot('elapsed_time', '[{}] Elapsed time ({})'.format(self._name, mode),
                       'Time (ms)'),
                 Bplot('sys_time', '[{}] System time ({})'.format(self._name, mode),
                       'Time (ms)'),
                 Bplot('user_time', '[{}] User time ({})'.format(self._name, mode),
                       'Time (ms)'),]
        correctors_ = self.correctors
        for bp in plots:
            for i, case in enumerate(cases):
                if corrected and self.get_case_id(case) in correctors_:
                    continue
                df = self.to_df(case=case, raw_header=True, corrected=corrected)
                repeat = df['i_th'].values.max() + 1
                if x is None:
                    x = range(1, len(self.step_col_df(self.get_case_id(case)))//repeat+1)
                kw = {'label': case}
                if plot_opt == 'all':
                    
                    for r in range(repeat):
                        dfq = df.query('i_th=={}'.format(r))
                        y = dfq[bp.key].values
                        plt.plot(x, y, customized_colors[i], **kw)
                        kw = {}
                elif plot_opt == 'mean':
                    y = df.groupby(['step'])[bp.key].mean().values
                    plt.plot(x, y, customized_colors[i], **kw)
                elif plot_opt == 'min':
                    y = df.groupby(['step'])[bp.key].min().values
                    plt.plot(x, y, customized_colors[i], **kw)
                elif plot_opt == 'max':
                    y = df.groupby(['step'])[bp.key].max().values
                    plt.plot(x, y, customized_colors[i], **kw)
                    
            plt.title(bp.title)
            plt.ylabel(bp.ylabel)
            plt.xlabel(self._step_header)
            plt.legend()
            plt.show()

    def prof_stats(self, case, step='first', measurement=0):
        tbl = self.env.measurement_tbl
        df = self.to_df(case, with_mem=False, with_times=False,
              with_step_info=True, corrected=False, raw_header=True, all_columns=True)
        if step=='last' or step==-1:
            step_ = df['step'].iloc[-1]
        elif step=='first' or step==0:
            step_ = df['step'].iloc[0]
        else:
            step_ = step
        case_id = self.get_case_id(case)
        stmt = (tbl.select().with_only_columns([tbl.c.prof]).
                    # values are casted to int here because
                    # np.int64 is not int in PY3!
                    where(tbl.c.case_id==int(case_id)).
                    where(tbl.c.i_th==int(measurement)).                                
                    where(tbl.c.step==int(step_)) 
                )
        with self.env.engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
        # TODO: use a REAL tmp file
        tmp_file_name = '/tmp/benchmarkit_out.prof'
        with open(tmp_file_name, 'wb') as tmp_file:
            tmp_file.write(row[0])
        # snakeviz is launched this way for virtualenv/anaconda compatibility
        c_opt = 'import sys, snakeviz.cli;sys.exit(snakeviz.cli.main())'
        cmd_ = [sys.executable, '-c', c_opt, tmp_file_name]
        p = subprocess.Popen(cmd_)
        time.sleep(3)
        p.terminate()
    @property
    def name(self):
        return self._name

    @property
    def env(self):
        if self._env is None:
            self._env = BenchEnv()
        return self._env

    @property
    def loop_var_proc(self):
        return self._loop_var_proc

    @property
    def time_flag(self):
        return self.with_time_prof

    @property
    def mem_flag(self):
        return self.with_memory_prof

    @property
    def prof_flag(self):
        return self.with_code_prof

    @property
    def repeat(self):
        return self.nb_repeat
    def runTest(self):
        self.runBench()


            

class BenchRunner():
    def __init__(self, func, case_id, corrected_by, bench):
        self._func = func #partial(func, bench)
        self._case_id = case_id
        self._corrected_by = corrected_by
        self._bench = bench
        self._args = None
        self._kwargs = None
        
    @property
    def bench(self):
        return self._bench
    
    @property
    def env(self):
        return self._bench.env
    def run_times(self, args, kwargs, step, i_th):
        ut, st, cut, cst, et = os.times()
        self._func(*args, **kwargs)
        ut_, st_, cut_, cst_, et_ = os.times()
        elapsed_time = et_ - et
        sys_time = st_ -st
        user_time = ut_ - ut
        ld_avg_1, ld_avg_5, ld_avg_15 = os.getloadavg()
        ## engine = create_engine('sqlite:///' + self.env.db_name, echo=True)
        ## metadata = MetaData()
        ## metadata.reflect(bind=engine)
        ## measurement_tbl = metadata.tables['measurement_tbl']
        stmt = (self.env.measurement_tbl.update().
                where(self.env.measurement_tbl.c.case_id==self._case_id).
                where(self.env.measurement_tbl.c.i_th==i_th).                
                where(self.env.measurement_tbl.c.step==step).
                values(
                    elapsed_time=elapsed_time,
                    sys_time=sys_time,
                    user_time=user_time,
                    ld_avg_1=ld_avg_1,
                    ld_avg_5=ld_avg_5,
                    ld_avg_15=ld_avg_15,
                    step_info=self.bench.step_info,
                    )
                )
        with self.env.engine.connect() as conn:
             conn.execute(stmt)
             
    def run_mem(self, args, kwargs, step, i_th):
        mem = memory_usage((self._func, args, kwargs), max_usage=True)[0]
        stmt = (self.env.measurement_tbl.update().
                where(self.env.measurement_tbl.c.case_id==self._case_id).
                where(self.env.measurement_tbl.c.i_th==i_th).                                
                where(self.env.measurement_tbl.c.step==step).
                values(
                    mem_usage=mem,
                    step_info=self.bench.step_info,                    
                    )
                )
        with self.env.engine.connect() as conn:
             conn.execute(stmt)
             
    def run_prof(self, args, kwargs, step, i_th):
        def to_profile():
            self._func(*args, **kwargs)
        # TODO: use a REAL tmp file            
        tmp_file_name = '/tmp/benchmarkit.prof'
        cProfile.runctx('to_profile()', globals(), locals(), tmp_file_name)
        with open(tmp_file_name, 'rb') as tmp_file:
            prof_blob = tmp_file.read()
        stmt = (self.env.measurement_tbl.update().
                where(self.env.measurement_tbl.c.case_id==self._case_id).
                where(self.env.measurement_tbl.c.i_th==i_th).                                
                where(self.env.measurement_tbl.c.step==step).
                values(
                    prof=prof_blob,
                    step_info=self.bench.step_info,                    
                    )
                )
        with self.env.engine.connect() as conn:
             conn.execute(stmt)

    def run(self, t):
        #args, kwargs = self.bench.input_proc.to_args(t)
        args, kwargs = (self.bench,), {}
        #values_ = self.bench.input_proc.to_dict(t)
        #v = self.bench.input_proc.to_value(t)
        step = self.bench.current_step
        values_ = dict(case_id=self._case_id, step=step)
        l_val = [dict(i_th=i) for i in six.moves.range(self.bench.repeat)]
        #map(lambda d: d.update(values_), l_val)
        for d in l_val:
            d.update(values_)
        ins = self.env.measurement_tbl.insert() #.values(**values_)
        with self.env.engine.connect() as conn:
            conn.execute(ins, l_val)
        for i_th in six.moves.range(self.bench.repeat):
            if self.bench.time_flag:
                p = Process(target=BenchRunner.run_times, args=(self, args, kwargs, step, i_th))
                p.start()
                p.join()
            #self.run_times(args, kwargs, v)
            if self.bench.mem_flag:
                p = Process(target=BenchRunner.run_mem, args=(self, args, kwargs, step, i_th))
                p.start()
                p.join()
            if self.bench.prof_flag:
                p = Process(target=BenchRunner.run_prof, args=(self, args, kwargs, step, i_th))
                p.start()
                p.join()

        
def banner(s, c='='):
    hr = c*(len(s) + 2) + '\n'
    s2 = ' ' + s + ' \n'
    return hr + s2 + hr

def print_banner(s, c='='):
    print(banner(s, c))
    
