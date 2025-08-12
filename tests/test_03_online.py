# type: ignore
import functools
import importlib
import inspect
import math
import random
import statistics
from typing import (
    Callable,
    Iterator,
    Any
)
import numpy as np
import pytest
from scipy import stats as sp_stats

from progressivis import stats


def load_stats() -> Iterator[Any]:
    for _, obj in inspect.getmembers(
            importlib.import_module("progressivis.stats"),
            inspect.isclass):
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
        (stats.Var(), np.var),
        (stats.Var(ddof=1), functools.partial(np.var, ddof=1)),
    ],
)
def test_univariate(stat: stats.Univariate,
                    func: Callable[[np.typing.ArrayLike], float]) -> None:
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
def test_bivariate(stat: stats.Bivariate,
                   func: Callable[[np.typing.ArrayLike,
                                   np.typing.ArrayLike], float]) -> None:
    X = [random.random() for _ in range(30)]
    Y = [random.random() * x for x in X]

    for i, (x, y) in enumerate(zip(X, Y)):
        stat.update(x, y)
        if i >= 1:
            assert math.isclose(
                stat.get(),
                func(X[: i + 1], Y[: i + 1]),
                abs_tol=1e-10
            )


@pytest.mark.parametrize(
    "stat",
    filter(
        lambda stat: hasattr(stat, "update_many")
        and issubclass(stat.__class__, stats.online.Univariate),
        load_stats(),
    ),
    ids=lambda stat: stat.__class__.__name__,
)
def test_update_many_univariate(stat: stats.Univariate) -> None:
    batch_stat = stat.clone()
    Y = np.random.random(50)

    for X in np.split(Y, 10):
        batch_stat.update_many(X)
        for x in X:
            stat.update(x)

    assert math.isclose(batch_stat.get(), stat.get())
    if hasattr(stat, "mean_ci_central_limit"):
        assert math.isclose(
            batch_stat.mean_ci_central_limit(),
            stat.mean_ci_central_limit()
        )
        ci = sp_stats.t.ppf(1-0.05, len(Y)-1) * np.std(Y) / np.sqrt(len(Y))
        assert math.isclose(
            stat.mean_ci_central_limit(), float(ci)
        )


@pytest.mark.parametrize(
    "stat",
    filter(
        lambda stat: hasattr(stat, "update_many")
        and issubclass(stat.__class__, stats.online.Bivariate),
        load_stats(),
    ),
    ids=lambda stat: stat.__class__.__name__,
)
def test_update_many_bivariate(stat: stats.Bivariate) -> None:
    batch_stat = stat.clone()

    for _ in range(5):
        X = np.random.random(30)
        Y = np.random.random(30)
        batch_stat.update_many(X, Y)
        for x, y in zip(X, Y):
            stat.update(x, y)
    assert math.isclose(batch_stat.get(), stat.get())


def _correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def test_update_many_covariance_matrix() -> None:
    mat = np.random.rand(3, 30)
    chunks = [0, 10, 20, 30]

    cm = stats.CovarianceMatrix()
    for c in range(1, len(chunks)):
        X = {col: mat[col, chunks[c-1]:chunks[c]]
             for col in range(mat.shape[0])}
        cm.update_many(X)
        submat = mat[:, 0:chunks[c]]
        assert np.allclose(cm.cov_matrix(), np.cov(submat))
        assert np.allclose(cm.corr_matrix(), np.corrcoef(submat))
