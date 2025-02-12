# type: ignore
from __future__ import annotations

from threading import Thread
from queue import Queue, ShutDown
from time import sleep
from timeit import default_timer

import pandas as pd
import numpy as np

TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"

def ignore(x):
    return x

def threaded_read_csv(
        reader: pd.io.parsers.readers.TextFileReader,
        queue: Queue):
    print("Starting thread")
    for df in reader:
        queue.put(df)
    queue.shutdown()

def test(filename, chunksize=100000):
    reader = pd.read_csv(filename, index_col=False, chunksize=chunksize)
    q = Queue()
    thread = Thread(target=threaded_read_csv, args=(reader, q))
    thread.start()
    while True:
        print("Waiting for next df")
        df = q.get()
        ignore(df)
        q.task_done()


class ThreadedReadCSV(Thread):
    def __init__(self, reader: pd.io.parsers.readers.TextFileReader, quantum=0.5):
        super().__init__(daemon=True)
        self.reader = reader
        self.queue = Queue()
        self.quantum = quantum
        self.times = np.zeros(5, dtype=np.float32)
        self.counts = np.zeros(5, dtype=np.int32)
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def _compute_steps(self, last_time: float, steps: int) -> int:
        self.times = np.roll(self.times, 1)
        self.times[0] = default_timer() - last_time
        self.counts = np.roll(self.counts, 1)
        self.counts[0] = steps
        count = self.counts.sum()
        time = self.times.sum()
        if time > 0:
            return int(self.quantum * count / time)
        return self.reader.chunksize

    def run(self) -> None:
        last = default_timer()
        for df in self.reader:
            if self.terminated:
                break  # Behave as if the stream was closed
            # print(f"loaded {id(df)}")
            self.queue.put(df)
            self.reader.chunksize = self._compute_steps(last, len(df))
            last = default_timer()

        self.queue.shutdown()

    def next(self) -> list[pd.DataFrame] | None:
        ret: list[pd.DataFrame] = []
        q = self.queue
        while True:
            try:
                df = q.get()
            except ShutDown:
                # Properly terminates the Queue and Thread
                q.join()
                self.join()
                return None
            ret.append(df)
            q.task_done()
            if q.empty():
                break
        return ret

def test2(filename, initialchunksize=1000):
    start = default_timer()
    total = 0
    reader = pd.read_csv(filename, index_col=False, chunksize=initialchunksize)
    t = ThreadedReadCSV(reader)
    print(f"Initialization took {default_timer()-start}s")
    start = default_timer()
    t.start()
    while True:
        sleep(1)
        dfs = t.next()
        if dfs is None:
            break
        sum = 0
        for df in dfs:
            sum += len(df)
            total += sum
        print(f"Read {len(dfs)} chunks of {sum}/{total} items")
    print(f"Done reading {total} lines in {default_timer()-start}s")

def test3(filename, initialchunksize=1000):
    start = default_timer()
    total = 0
    reader = pd.read_csv(filename, index_col=False, chunksize=initialchunksize)
    t = ThreadedReadCSV(reader)
    print(f"Initialization took {default_timer()-start}s")
    start = default_timer()
    t.start()
    while True:
        sleep(1)
        dfs = t.next()
        if dfs is None:
            break
        sum = 0
        for df in dfs:
            sum += len(df)
            total += sum
        print(f"Read {len(dfs)} chunks of {sum}/{total} items")
        if total > 1_000_000:
            print("Terminating")
            t.terminate()
    print(f"Done reading {total} lines in {default_timer()-start}s")


if __name__ == "__main__":
    test2(TAXI_FILE)
