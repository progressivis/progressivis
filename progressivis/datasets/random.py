import numpy as np
import csv
import os
import os.path

# filename='data/bigfile.csv'
# rows = 1000000
# cols = 30

def generate_random_csv(filename, rows, cols, seed=1234):
    if os.path.exists(filename):
        return filename
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            np.random.seed(seed=seed)
            for _ in range(0,rows):
                row=list(np.random.normal(loc=0.5, scale=0.8, size=cols))
                writer.writerow(row)
    except (KeyboardInterrupt, SystemExit):
        os.remove(filename)
        raise
    return filename
