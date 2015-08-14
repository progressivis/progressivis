import numpy as np
import csv
import os

# filename='data/bigfile.csv'
# rows = 1000000
# cols = 30

def generate_random_csv(filename, rows, cols):
    if not os.path.exists(filename):
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            np.random.seed(seed=1234)
            for r in range(0,rows):
                row=list(np.random.rand(cols))
                writer.writerow(row)
    return filename

