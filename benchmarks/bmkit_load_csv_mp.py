import pandas as pd
import csv
from progressivis import Scheduler
from progressivis.io import CSVLoader
from benchmarkit import BenchMarkIt
import sys
import os
import os.path
import subprocess
import glob
import sqlite3
from collections import OrderedDict
from multiprocessing import Process
import dask.dataframe as dd


def p10s_read_csv(f):
    s = Scheduler()
    module = CSVLoader(f, header=None, scheduler=s)
    _ = module
    s.start()


def none_read_csv(f):
    res = {}
    with open(f, "rb") as csvfile:
        for _ in csvfile:
            pass
    return res


def naive_read_csv(f):
    res = {}
    with open(f, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        first = next(reader)
        for i, cell in enumerate(first):
            res[i] = [float(cell)]
        for row in reader:
            for i, cell in enumerate(row):
                res[i].append(float(cell))
    return res


def dask_read_csv(f):
    dd.read_csv(f).compute()


def cleanup_hdf5():
    for d in glob.glob("/tmp/progressivis_*"):
        subprocess.call(["/bin/rm", "-rf", d])


func_dict = OrderedDict(
    [
        ("Empty", none_read_csv),
        ("Pandas", pd.read_csv),
        ("Dask", dask_read_csv),
        ("Progressivis", p10s_read_csv),
        ("Naive", naive_read_csv),
    ]
)


def bmkit_worker(csv_file, db_name, label, mode, nb_lines):
    func = func_dict[label]
    mem_flag = True if mode == "mem" else False
    time_flag = True if mode == "time" else False
    bm = BenchMarkIt(func, [csv_file], label=label)
    bm.run(tm=time_flag, mem=mem_flag)
    d = bm.to_dict()
    print(d)
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    if mem_flag:
        c.execute(
            "UPDATE {}_tbl SET memory=? WHERE mega_lines=?".format(label.lower()),
            (
                d["memory"],
                nb_lines,
            ),
        )
    if time_flag:
        c.execute(
            """UPDATE {}_tbl SET elapsed_time=?,
        sys_time=?, user_time=? WHERE mega_lines=?""".format(
                label.lower()
            ),
            (
                d["elapsed_time"],
                d["sys_time"],
                d["user_time"],
                nb_lines,
            ),
        )
    conn.commit()
    conn.close()
    cleanup_hdf5()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage {} <dbname> <csvfile1> [<csvfile2>...<csvfileN>]".format(sys.argv[0])
        )
        sys.exit()
    db_name = sys.argv[1]
    labels = func_dict.keys()  # ["Empty", "Pandas", "Progressivis", "Naive"]
    low_labels = [e.lower() for e in labels]
    if os.path.exists(db_name):
        print("Database {} already exists, exit".format(db_name))
        sys.exit()
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    for llab in low_labels:
        c.execute(
            """CREATE TABLE {}_tbl
        (mega_lines integer, memory real, elapsed_time real,
        sys_time real, user_time real)""".format(
                llab
            )
        )
    csv_file_dict = {}
    for csv_file in sys.argv[2:]:
        nb_lines, _ = subprocess.check_output(["wc", "-l", csv_file]).split(" ")
        nb_mega = int(nb_lines) // 1000000
        csv_file_dict[csv_file] = nb_mega
        for llab in low_labels:
            c.execute(
                "INSERT INTO {}_tbl VALUES(?, ?, ?, ?, ?)".format(llab),
                (nb_mega, None, None, None, None),
            )
    conn.commit()
    conn.close()
    for csv_file in sys.argv[2:]:
        for lab in labels:
            p = Process(
                target=bmkit_worker,
                args=(csv_file, db_name, lab, "time", csv_file_dict[csv_file]),
            )
            p.start()
            p.join()
            p = Process(
                target=bmkit_worker,
                args=(csv_file, db_name, lab, "mem", csv_file_dict[csv_file]),
            )
            p.start()
            p.join()
