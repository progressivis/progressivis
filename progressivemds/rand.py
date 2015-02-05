# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def random_mds(n):
    x = np.random.rand(n)
    y = np.random.rand(n)
    label = ["i%d" % i for i in range(n)]
    return pd.DataFrame({'label': label,
                         'x': x,
                         'y': y,
                         'color': "black" })

def test():
    random_mds(100).to_csv("../mds.tsv", sep='\t', header=True, index=True, index_label="id", encoding="utf-8")

if __name__ == '__main__':
    test()

