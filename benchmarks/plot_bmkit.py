import sqlite3
import pandas as pd
import sys
from collections import namedtuple

if len(sys.argv) != 2:
    print("Usage: {} db-name".format(sys.argv[0]))
    sys.exit(1)
db_name = sys.argv[1]
conn = sqlite3.connect(db_name)
import matplotlib.pyplot as plt

df_empty = pd.read_sql_query("SELECT * from empty_tbl", conn)
df_pandas = pd.read_sql_query("SELECT * from pandas_tbl", conn)
df_dask = pd.read_sql_query("SELECT * from dask_tbl", conn)
df_p10s = pd.read_sql_query("SELECT * from progressivis_tbl", conn)
df_naive =  pd.read_sql_query("SELECT * from naive_tbl", conn)

x = df_empty["mega_lines"].values

Bplot = namedtuple('Bplot','key, title, ylabel')

plots = [Bplot('memory', 'Memory usage', 'Used memory (Mb)'),
         Bplot('elapsed_time', 'Elapsed time', 'Time (ms)'),
         Bplot('sys_time', 'System time', 'Time (ms)'),
         Bplot('user_time', 'User time', 'Time (ms)'),]

for bp in plots:
    y_empty = df_empty[bp.key].values
    y_pandas = df_pandas[bp.key].values
    y_dask = df_dask[bp.key].values    
    y_p10s = df_p10s[bp.key].values
    y_naive = df_naive[bp.key].values

    plt.plot(x, y_empty, 'b--', label='Empty')
    plt.plot(x, y_pandas, 'r--', label='Pandas')
    plt.plot(x, y_dask, 'm--', label='Dask')    
    plt.plot(x, y_p10s, 'g--', label='Progressivis')
    plt.plot(x, y_naive, 'y--', label='Naive')

    plt.title(bp.title)
    plt.ylabel(bp.ylabel)
    plt.xlabel('million CSV lines in the file (x 30 cols)')
    plt.legend()
    plt.show()


