# type: ignore
from __future__ import annotations

from threading import Thread
try:
    from queue import Queue, ShutDown
except ImportError:
    pass
from timeit import default_timer

import pandas as pd
import numpy as np

TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"


class ThreadedReadCSV(Thread):
    def __init__(self, reader: pd.io.parsers.readers.TextFileReader, input_stream, quantum=0.5):
        super().__init__(daemon=True)
        self.reader = reader
        self.queue = Queue()
        self._input_stream = input_stream
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
            self.queue.put((self._input_stream.tell(), df))
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
