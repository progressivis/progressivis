import sqlite3
import pandas as pd
import sys

def print_box(text):
    stars = "*" * (len(text) + 4)
    print(stars)
    print("* "+text+" *")
    print(stars)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} db-name".format(sys.argv[0]))
        sys.exit(1)
    
    db_name = sys.argv[1]
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * from empty_tbl", conn)
    print_box("Blind csv load")
    print(df)
    df = pd.read_sql_query("SELECT * from pandas_tbl", conn)
    print_box("Pandas csv load")
    print(df)
    df = pd.read_sql_query("SELECT * from dask_tbl", conn)
    print_box("Dask csv load")
    print(df)
    df = pd.read_sql_query("SELECT * from progressivis_tbl", conn)
    print_box("Progressivis csv load")
    print(df)
    df = pd.read_sql_query("SELECT * from naive_tbl", conn)
    print_box("Naive csv load")
    print(df)


