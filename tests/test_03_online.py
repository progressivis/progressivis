import functools
import importlib
import inspect
import math
import random
import statistics

import numpy as np
import pytest
from scipy import stats as sp_stats

from progressivis import stats


def load_stats():
    for _, obj in inspect.getmembers(importlib.import_module("progressivis.stats"), inspect.isclass):
        try:
            if inspect.isabstract(obj):
                continue

            sig = inspect.signature(obj)
            yield obj(
                **{
                    param.name: param.default if param.default != param.empty else 1
                    for param in sig.parameters.values()
                }
            )
        except ValueError:
            yield obj()


@pytest.mark.parametrize(
    "stat, func",
    [
        (stats.Mean(), statistics.mean),
        (stats.Var(ddof=0), np.var),
        (stats.Var(), functools.partial(np.var, ddof=1)),
    ],
)
def test_univariate(stat, func):
    X = [random.random() for _ in range(30)]

    for i, x in enumerate(X):
        stat.update(x)
        if i >= 1:
            assert math.isclose(stat.get(), func(X[: i + 1]), abs_tol=1e-10)


@pytest.mark.parametrize(
    "stat, func",
    [
        (stats.Cov(), lambda x, y: np.cov(x, y)[0, 1]),
        (stats.Corr(), lambda x, y: sp_stats.pearsonr(x, y)[0]),
    ],
)
def test_bivariate(stat, func):
    X = [random.random() for _ in range(30)]
    Y = [random.random() * x for x in X]

    for i, (x, y) in enumerate(zip(X, Y)):
        stat.update(x, y)
        if i >= 1:
            assert math.isclose(stat.get(), func(X[: i + 1], Y[: i + 1]), abs_tol=1e-10)


@pytest.mark.parametrize(
    "stat",
    filter(
        lambda stat: hasattr(stat, "update_many")
        and issubclass(stat.__class__, stats.online.Univariate),
        load_stats(),
    ),
    ids=lambda stat: stat.__class__.__name__,
)
def test_update_many_univariate(stat):
    batch_stat = stat.clone()

    for _ in range(5):
        X = np.random.random(10)
        batch_stat.update_many(X)
        for x in X:
            stat.update(x)

    assert math.isclose(batch_stat.get(), stat.get())


@pytest.mark.parametrize(
    "stat",
    filter(
        lambda stat: hasattr(stat, "update_many")
        and issubclass(stat.__class__, stats.online.Bivariate),
        load_stats(),
    ),
    ids=lambda stat: stat.__class__.__name__,
)
def test_update_many_bivariate(stat):
    batch_stat = stat.clone()

    for _ in range(5):
        X = np.random.random(10)
        Y = np.random.random(10)
        batch_stat.update_many(X, Y)
        for x, y in zip(X, Y):
            stat.update(x, y)

    assert math.isclose(batch_stat.get(), stat.get())
