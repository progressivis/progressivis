import numpy as np
import pandas as pd
import six

from memory_profiler import memory_usage
import sys, os, time
from collections import OrderedDict, Iterable, namedtuple
from sqlalchemy import (Table, Column, Integer, Float, String, Sequence, BLOB,
                        MetaData, ForeignKey, create_engine, select,
                        UniqueConstraint)
#from multiprocessing import Process
import matplotlib
import matplotlib.pyplot as plt
import numpy.distutils.cpuinfo as cpuinfo
import platform
import json
import cProfile
import subprocess, time

LoopVarDesc = namedtuple("LoopVarDesc", "type, title, desc, func")
# lvd = LoopVarDesc(type=str,title="X", desc="Blah, blah", func=lambda x: x.upper())

multi_processing = True # use False only for debug!

if multi_processing:
    from multiprocessing import Process
else:
    class Process(object):
        def __init__(self, target, args):
            target(*args)
        def start(self):
            pass
        def join(self):
            pass
        
def get_cpu_info():
    return json.dumps(cpuinfo.cpu.info)


def table_exists(tbl, conn):
    cur = conn.cursor()
    return list(cur.execute(
        """SELECT name FROM sqlite_master WHERE type=? AND name=?""",
        ('table', tbl)))

def describe_table(tbl, conn):
    cur = conn.cursor()
    return list(c.execute("PRAGMA table_info([{}])".format(tbl)))

def dump_table(table, db_name):
    #lst_table=['bench_tbl', 'case_tbl','measurement_tbl']
    engine = create_engine('sqlite:///' + db_name, echo=False)
    metadata = MetaData(bind=engine)
    tbl = Table(table, metadata, autoload=True)
    conn = engine.connect()
    #print(metadata.tables[tbl])
    s = tbl.select() #select([[tbl]])
    df =  pd.read_sql_query(s, conn)
    print(df)


    
class BenchEnv(object):
    def __init__(self, db_name, append_mode=True):
        self._db_name = db_name
        self._engine = create_engine('sqlite:///' + db_name, echo=False)
        self._metadata = MetaData(bind=self._engine)
        self._bench_tbl = None
        self._case_tbl = None
        self._measurement_tbl = None
        self._append_mode = append_mode
        self.create_tables_if()

    @property
    def engine(self):
        return self._engine

    @property
    def db_name(self):
        return self._db_name

    @property
    def bench_tbl(self):
        return self._bench_tbl

    @property
    def measurement_tbl(self):
        return self._measurement_tbl
    @property
    def bench_list(self):
        tbl = self._bench_tbl
        s = (select([tbl]).with_only_columns([tbl.c.name]))
        with self.engine.connect() as conn:
            rows = conn.execute(s).fetchall()
            return [e[0] for e in rows] 

    def create_tables_if(self):
        if 'bench_tbl' in self.engine.table_names():
            self._bench_tbl = Table('bench_tbl', self._metadata, autoload=True)
            self._case_tbl = Table('case_tbl', self._metadata, autoload=True)
            self._measurement_tbl = Table('measurement_tbl', self._metadata, autoload=True)
            return
        self._bench_tbl = Table('bench_tbl', self._metadata,
                                Column('id', Integer, Sequence('user_id_seq'), primary_key=True),                                
                                Column('name', String, unique=True),
                                Column('description', String),
                                Column('py_version', String),
                                Column('py_compiler', String),
                                Column('py_platform', String),
                                Column('py_impl', String),                                
                                Column('cpu_info', String),
                                Column('repr_type', String),
                                Column('user_col_label', String),                                                                

                                autoload=False)
        self._case_tbl = Table('case_tbl', self._metadata,
                               Column('id', Integer, Sequence('user_id_seq'), primary_key=True),                                
                               Column('name', String, unique=True),
                               Column('bench_id', Integer, ForeignKey('bench_tbl.id')),
                               Column('corrected_by', Integer), # TODO: ref integrity
                               Column('description', String),
                               UniqueConstraint('name', 'bench_id', name='uc1'),
                               autoload=False)
        self._measurement_tbl = Table('measurement_tbl', self._metadata,
                                      Column('id', Integer, Sequence('user_id_seq'), primary_key=True),
                                      
                                      Column('case_id', Integer),
                                      Column('i_th', Integer),                                      
                                      Column('user_col_str', String),
                                      Column('user_col_int',  Integer),
                                      Column('user_col_float',  Float),
                                      Column('mem_usage',  Float),
                                      Column('elapsed_time',  Float),
                                      Column('sys_time',  Float),
                                      Column('user_time',  Float),
                                      Column('ld_avg_1',  Float),
                                      Column('ld_avg_5',  Float),
                                      Column('ld_avg_15',  Float),
                                      Column('prof',  BLOB),                                                                                                                  
                                       autoload=False)
    
        self._metadata.create_all(self._engine, checkfirst=self._append_mode)

def default_loop_var_proc(loop_var):
    args = loop_var if isinstance(loop_var, tuple) else (loop_var,)
    cols = {}
    if isinstance(loop_var, (six.integer_types, np.integer)):
        cols['user_col_int'] = int(loop_var)
    elif isinstance(loop_var, float):
        cols['user_col_float'] = float(loop_var)
    else:
        cols['user_col_str'] = str(loop_var)
    return args, {}, cols


# InputProc = namedtuple("InputProc", "input_type, label, desc, inp_to_args, inp_to_col")
# lvd = LoopVarDesc(type=str,title="X", desc="Blah, blah", func=lambda x: x.upper())

class InputProc(object):
    def __init__(self, label, repr_type=str):
        if repr_type not in (int, float, str):
            raise ValueError("{} type not allowed".format(repr_type))
        type_dict = {float: 'user_col_float', int: 'user_col_int',
                     str: 'user_col_str'}
        self._repr_type = repr_type
        self._user_col = type_dict[repr_type]
        self._label = label
    def to_args(self, inp):
        return (self._repr_type(inp),), {} 
    def to_value(self, inp):
        return self._repr_type(inp)
    def to_dict(self, inp):
        return {self.user_col: self.to_value(inp)}
    @property
    def user_col(self):
        return self._user_col
    @property
    def label(self):
        return self._label

class BenchmarkIt(object):
    def __init__(self, env, name, loop, repeat=1, desc="",
                 input_proc=InputProc(repr_type=str, label='UserCol'),
                 time_bm=True, memory_bm=True, prof=True, after_loop_func=None):
        self._env = env
        self._name = name
        self._loop = loop
        self._repeat = repeat
        self._desc = desc
        self._input_proc = input_proc
        self._time_flag = time_bm
        self._mem_flag = memory_bm
        self._prof_flag = prof
        self._after_loop_func = after_loop_func
        self._sql_id = None
        self._cases = []
        self._corrections = []
    @staticmethod
    def load(env, name):
        bm =  BenchmarkIt(env, name, -1)
        tbl = env._bench_tbl
        s = (select([tbl]).
             where(tbl.c.name==bm._name))
        with env.engine.connect() as conn:
            row = conn.execute(s).fetchone()
            row = dict(row)
            repr_type = {'str':str, 'int':int, 'float':float}[row['repr_type']]
            label  = row['user_col_label']
            bm._input_proc = InputProc(repr_type=repr_type, label=label)
        return bm
    def __enter__(self):
        tbl = self._env._bench_tbl
        ins = tbl.insert().values(name=self._name, description=self._desc,
                                  py_version=platform.python_version(),
                                  py_compiler=platform.python_compiler(),
                                  py_platform=platform.platform(),
                                  py_impl=platform.python_implementation(),
                                  cpu_info=get_cpu_info(),
                                  repr_type=self._input_proc._repr_type.__name__,
                                  user_col_label=self._input_proc._label                                
                                  )
        with self.env.engine.connect() as conn:
            conn.execute(ins)
        ## s = (select([tbl]).with_only_columns([tbl.c.id]).
        ##      where(tbl.c.name==self._name))
        ## #http://www.rmunn.com/sqlalchemy-tutorial/tutorial.html
        ## with self.env.engine.connect() as conn:
        ##     row = conn.execute(s).fetchone()
        ##     self._sql_id = row[0]
        return self
    @property
    def sql_id(self):
        if self._sql_id is not None:
            return self._sql_id
        tbl = self._env._bench_tbl
        s = (select([tbl]).with_only_columns([tbl.c.id, tbl.c.name]).
             where(tbl.c.name==self._name))
        #http://www.rmunn.com/sqlalchemy-tutorial/tutorial.html
        with self.env.engine.connect() as conn:
            row = conn.execute(s).fetchone()
            self._sql_id = row[0]
            return self._sql_id
    def __call__(self, case, corrected_by=None):
        self._corrections.append((case, corrected_by))
        tbl = self._env._case_tbl
        ins = tbl.insert().values(name=case, bench_id=self.sql_id)
        with self.env.engine.connect() as conn:
            conn.execute(ins)
        ## s = (select([tbl]).with_only_columns([tbl.c.id]).
        ##      where(tbl.c.name==self._name))
        ## with self.env.engine.connect() as conn:
        ##     row = conn.execute(s).fetchone()
        ##     case_id = row[0]
        case_id = self.get_case_id(case)
        def fun(func):
            runner = BenchRunner(func, case_id, corrected_by, self)
            self._cases.append(runner)
        return fun
    def _update_corrections(self):
        for case, corr_by in self._corrections:
            if corr_by is None:
                continue
            case_id = self.get_case_id(case)
            corr_id = self.get_case_id(corr_by)            
            tbl = self._env._case_tbl
            stmt = (tbl.update().where(tbl.c.id==case_id).
                    values(corrected_by=corr_id))
            with self.env.engine.connect() as conn:
                conn.execute(stmt)

    def get_case_id(self, case):
        tbl = self._env._case_tbl
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
        tbl = self._env._case_tbl
        s = (select([tbl]).with_only_columns([tbl.c.corrected_by]).
             where(tbl.c.id==case_id))
        with self.env.engine.connect() as conn:
            row = conn.execute(s).fetchone()
            return row[0]
    @property
    def correctors(self):
        tbl = self._env._case_tbl
        s = select([tbl]).with_only_columns([tbl.c.corrected_by])
        with self.env.engine.connect() as conn:
            rows = conn.execute(s).fetchall()
            return [e[0] for e in rows if e[0] is not None] 

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value:
            raise
        #self.create_tables_if()
        self._update_corrections()
        if isinstance(self._loop, Iterable):
            loop_ = self._loop
        if isinstance(self._loop, (six.integer_types, np.integer)):        
            loop_ = range(self._loop)
        for elt in self._cases:
            for arg in loop_:
                elt.run(arg)
            if callable(self._after_loop_func):
                self._after_loop_func()
                
    @property
    def _col_dict(self):
        return {'case': 'Case', 'corrected_by':'Corrected by', 'i_th':'Measure',
                'mem_usage': 'Memory usage', 'elapsed_time': 'Elapsed time',
                'sys_time': 'System time', 'user_time': 'User time',
                'ld_avg_1':'Load avg.(-1s)',
                'ld_avg_5':'Load avg.(-5s)','ld_avg_15':'Load avg.(-15s)',
                self.input_proc.user_col: self.input_proc.label}
    @property
    def case_names(self):
        tbl = self._env._case_tbl
        s = (select([tbl]).with_only_columns([tbl.c.name]).
             where(tbl.c.bench_id==self.sql_id))
        with self.env.engine.connect() as conn:
            rows = conn.execute(s).fetchall()
            return set([elt.name for elt in rows])
        #df =  pd.read_sql_query(s, conn)
        #return set(df['name'].values)

    def user_col_df(self, case_id):
        projection = [self.input_proc.user_col]
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
              with_user_col=True, corrected=True, raw_header=False):
        if isinstance(case, six.string_types):
            case_id = self.get_case_id(case)
        else:
            case_id = case
        projection = ['i_th']
        projection += [self.input_proc.user_col] if with_user_col==True else []
        if  with_mem:
            projection.append('mem_usage')
        if with_times:
            projection += ['elapsed_time', 'sys_time', 'user_time']
        projection += ['ld_avg_1', 'ld_avg_5', 'ld_avg_15']
        tbl = self.env.measurement_tbl
        only_columns = [col.name for col in tbl.columns if col.name in projection]
        s = (select([tbl]).with_only_columns(only_columns).
             where(tbl.c.case_id==case_id).
             order_by(tbl.c.id))
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
        plots = [Bplot('mem_usage', 'Memory usage ({})'.format(mode),
                       'Used memory (Mb)'),
                 Bplot('elapsed_time', 'Elapsed time ({})'.format(mode),
                       'Time (ms)'),
                 Bplot('sys_time', 'System time ({})'.format(mode),
                       'Time (ms)'),
                 Bplot('user_time', 'User time ({})'.format(mode),
                       'Time (ms)'),]
        correctors_ = self.correctors
        for bp in plots:
            for i, case in enumerate(cases):
                if corrected and self.get_case_id(case) in correctors_:
                    continue
                df = self.to_df(case=case, raw_header=True, corrected=corrected)
                repeat = df['i_th'].values.max() + 1
                if x is None:
                    #x = df[self.input_proc.user_col]
                    x = range(1, len(self.user_col_df(self.get_case_id(case)))//repeat+1)
                kw = {'label': case}
                if plot_opt == 'all':
                    
                    for r in range(repeat):
                        dfq = df.query('i_th=={}'.format(r))
                        y = dfq[bp.key].values
                        plt.plot(x, y, customized_colors[i], **kw)
                        kw = {}
                elif plot_opt == 'mean':
                    y = df.groupby([self.input_proc.user_col])[bp.key].mean().values
                    plt.plot(x, y, customized_colors[i], **kw)
                elif plot_opt == 'min':
                    y = df.groupby([self.input_proc.user_col])[bp.key].min().values
                    plt.plot(x, y, customized_colors[i], **kw)
                elif plot_opt == 'max':
                    y = df.groupby([self.input_proc.user_col])[bp.key].max().values
                    plt.plot(x, y, customized_colors[i], **kw)
                    
            plt.title(bp.title)
            plt.ylabel(bp.ylabel)
            plt.xlabel(self.input_proc.label)
            plt.legend()
            plt.show()

    def prof_stats(self, case, step='first', measurement=0):
        tbl = self.env.measurement_tbl
        df = self.to_df(case, with_mem=False, with_times=False,
              with_user_col=True, corrected=False, raw_header=True)
        if step=='last' or step==-1:
            step_ = df[self.input_proc.user_col].iloc[-1]
        elif step=='first' or step==0:
            step_ = df[self.input_proc.user_col].iloc[0]
        else:
            step_ = step
        case_id = self.get_case_id(case)
        stmt = (tbl.select().with_only_columns([tbl.c.prof]).
                where(tbl.c.case_id==case_id).
                where(tbl.c.i_th==measurement).                                
                where(tbl.c[self.input_proc.user_col]==step_)
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
        return self._env

    @property
    def loop_var_proc(self):
        return self._loop_var_proc

    @property
    def time_flag(self):
        return self._time_flag

    @property
    def mem_flag(self):
        return self._mem_flag

    @property
    def prof_flag(self):
        return self._prof_flag

    @property
    def repeat(self):
        return self._repeat

    @property
    def input_proc(self):
        return self._input_proc

            

class BenchRunner():
    def __init__(self, func, case_id, corrected_by, bench):
        self._func = func
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
    def run_times(self, args, kwargs, v, i_th):
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
        ##if self.bench.input_proc.user_col == 'user_col_str'
        stmt = (self.env.measurement_tbl.update().
                where(self.env.measurement_tbl.c.case_id==self._case_id).
                where(self.env.measurement_tbl.c.i_th==i_th).                
                where(self.env.measurement_tbl.c[self.bench.input_proc.user_col]==v).
                values(
                    elapsed_time=elapsed_time,
                    sys_time=sys_time,
                    user_time=user_time,
                    ld_avg_1=ld_avg_1,
                    ld_avg_5=ld_avg_5,
                    ld_avg_15=ld_avg_15,
                    )
                )
        with self.env.engine.connect() as conn:
             conn.execute(stmt)
             
    def run_mem(self, args, kwargs, v, i_th):
        mem = memory_usage((self._func, args, kwargs), max_usage=True)[0]
        stmt = (self.env.measurement_tbl.update().
                where(self.env.measurement_tbl.c.case_id==self._case_id).
                where(self.env.measurement_tbl.c.i_th==i_th).                                
                where(self.env.measurement_tbl.c[self.bench.input_proc.user_col]==v).
                values(
                    mem_usage=mem,
                    )
                )
        with self.env.engine.connect() as conn:
             conn.execute(stmt)
             
    def run_prof(self, args, kwargs, v, i_th):
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
                where(self.env.measurement_tbl.c[self.bench.input_proc.user_col]==v).
                values(
                    prof=prof_blob,
                    )
                )
        with self.env.engine.connect() as conn:
             conn.execute(stmt)

    def run(self, t):
        args, kwargs = self.bench.input_proc.to_args(t)
        values_ = self.bench.input_proc.to_dict(t)
        v = self.bench.input_proc.to_value(t)
        values_.update(dict(case_id=self._case_id))
        l_val = [dict(i_th=i) for i in six.moves.range(self.bench.repeat)]
        #map(lambda d: d.update(values_), l_val)
        for d in l_val:
            d.update(values_)
        ins = self.env.measurement_tbl.insert() #.values(**values_)
        with self.env.engine.connect() as conn:
            conn.execute(ins, l_val)
        for i_th in six.moves.range(self.bench.repeat):
            if self.bench.time_flag:
                p = Process(target=BenchRunner.run_times, args=(self, args, kwargs, v, i_th))
                p.start()
                p.join()
            #self.run_times(args, kwargs, v)
            if self.bench.mem_flag:
                p = Process(target=BenchRunner.run_mem, args=(self, args, kwargs, v, i_th))
                p.start()
                p.join()
            if self.bench.prof_flag:
                p = Process(target=BenchRunner.run_prof, args=(self, args, kwargs, v, i_th))
                p.start()
                p.join()

def banner(s, c='='):
    hr = c*(len(s) + 2) + '\n'
    s2 = ' ' + s + ' \n'
    return hr + s2 + hr

def print_banner(s, c='='):
    print(banner(s, c))
    
if __name__ == '__main__':
    import argparse
    import sys
    from tabulate import tabulate
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="measurements database", nargs=1,
                        required=True)
    ## parser.add_argument("--bench", help="bench to visualize", nargs=1,
    ##                     required=False)
    parser.add_argument("--cases", help="case(s) to visualize", nargs='+',
                        required=False)        
    parser.add_argument("--summary", help="List of cases in the DB",
                        action='store_const', const=42, required=False)
    parser.add_argument("--plot", help="Plot measurements", #action='store_const',
                        nargs=1, required=False, choices=['min', 'max', 'mean', 'all'])                     
    parser.add_argument("--pstats", help="Profile stats for: case[:first] | case:last | case:step", #action='store_const',
                        nargs=1, required=False)                     
    parser.add_argument("--no-corr", help="No correction", action='store_const',
                        const=1, required=False)                     
    args = parser.parse_args()
    db_name = args.db[0]
    if args.plot:
        plot_opt = args.plot[0]
    benv = BenchEnv(db_name=db_name)
    bench_name = benv.bench_list[0]
    if args.summary:
        bench = BenchmarkIt.load(env=benv, name=bench_name)
        print("List of cases: {}".format(", ".join(bench.case_names)))
        sys.exit(0)
    corr = True
    if args.no_corr:
        corr = False        
    bench = BenchmarkIt.load(env=benv, name=bench_name)
    if args.cases:
        cases = args.cases
    else:
        cases = bench.case_names
    print_banner("Cases: {}".format(" ".join(cases)))
    for case in cases:
        print_banner("Case: {}".format(case))
        df = bench.to_df(case=case, corrected=corr)
        print(tabulate(df, headers='keys', tablefmt='psql'))
    if args.plot:
        bench.plot(cases=cases, corrected=corr, plot_opt=plot_opt)
    #bench.prof_stats("P10sH5MinMax")
    #dump_table('measurement_tbl', 'prof.db')
    if args.pstats:
        case_step = args.pstats[0]
        if ':' not in case_step:
            case = case_step
            step = 'first'
        else:
            case, step = case_step.split(':', 1)
        bench.prof_stats(case, step)
