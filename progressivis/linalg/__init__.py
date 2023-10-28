# flake8: noqa
from .elementwise import (
    Unary,
    Binary,
    ColsBinary,
    Reduce,
    func2class_name,
    unary_module,
    make_unary,
    binary_module,
    make_binary,
    reduce_module,
    make_reduce,
    binary_dict_int_tst,
    unary_dict_gen_tst,
    binary_dict_gen_tst,
)
from .linear_map import LinearMap
from .nexpr import NumExprABC
from .mixufunc import make_local, make_local_dict, get_ufunc_args, MixUfuncABC
from ..core.module import document, PColumns, PColsList, PColsDict
from typing import Any, Optional, List
import numpy as np


@document
class Absolute(Unary):
    """
    Applies :meth:`numpy.absolute` over all input columns or over a subset
    """
    def __init__(self,
                 **kwds: Any):
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(np.absolute, **kwds)


@document
class Add(Binary):
    """
    Applies :meth:`numpy.add` over two sets of columns. One of them belongs to the
    ``first`` input table and the other belongs to the ``second``
    """
    def __init__(self,
                 **kwds: Any):
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(np.add, **kwds)


@document
class ColsAdd(ColsBinary):
    def __init__(self,
                 cols_out: Optional[List[str]] = None,
                 **kwds: Any):
        """
        Args:
            cols_out: denotes the names of columns in the ``result`` table
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(np.add, cols_out=cols_out, **kwds)

@document
class AddReduce(Reduce):
    """
    Applies :meth:`numpy.add.reduce` over all input columns or over a subset of them
    """
    def __init__(self,
                 **kwds: Any):
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(np.add, **kwds)


from ._elementwise import (
    BitwiseNot,
    # Absolute,
    Arccos,
    Arccosh,
    Arcsin,
    Arcsinh,
    Arctan,
    Arctanh,
    Cbrt,
    Ceil,
    Conj,
    Conjugate,
    Cos,
    Cosh,
    Deg2rad,
    Degrees,
    Exp,
    Exp2,
    Expm1,
    Fabs,
    Floor,
    Frexp,
    Invert,
    Isfinite,
    Isinf,
    Isnan,
    Isnat,
    Log,
    Log10,
    Log1p,
    Log2,
    LogicalNot,
    Modf,
    Negative,
    Positive,
    Rad2deg,
    Radians,
    Reciprocal,
    Rint,
    Sign,
    Signbit,
    Sin,
    Sinh,
    Spacing,
    Sqrt,
    Square,
    Tan,
    Tanh,
    Trunc,
    Abs,
    # Add,
    Arctan2,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Copysign,
    Divide,
    Divmod,
    Equal,
    FloorDivide,
    FloatPower,
    Fmax,
    Fmin,
    Fmod,
    Gcd,
    Greater,
    GreaterEqual,
    Heaviside,
    Hypot,
    Lcm,
    Ldexp,
    LeftShift,
    Less,
    LessEqual,
    Logaddexp,
    Logaddexp2,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Maximum,
    Minimum,
    Mod,
    Multiply,
    Nextafter,
    NotEqual,
    Power,
    Remainder,
    RightShift,
    Subtract,
    TrueDivide,
    # ColsAdd,
    ColsArctan2,
    ColsBitwiseAnd,
    ColsBitwiseOr,
    ColsBitwiseXor,
    ColsCopysign,
    ColsDivide,
    ColsDivmod,
    ColsEqual,
    ColsFloorDivide,
    ColsFloatPower,
    ColsFmax,
    ColsFmin,
    ColsFmod,
    ColsGcd,
    ColsGreater,
    ColsGreaterEqual,
    ColsHeaviside,
    ColsHypot,
    ColsLcm,
    ColsLdexp,
    ColsLeftShift,
    ColsLess,
    ColsLessEqual,
    ColsLogaddexp,
    ColsLogaddexp2,
    ColsLogicalAnd,
    ColsLogicalOr,
    ColsLogicalXor,
    ColsMaximum,
    ColsMinimum,
    ColsMod,
    ColsMultiply,
    ColsNextafter,
    ColsNotEqual,
    ColsPower,
    ColsRemainder,
    ColsRightShift,
    ColsSubtract,
    ColsTrueDivide,
    # AddReduce,
    Arctan2Reduce,
    BitwiseAndReduce,
    BitwiseOrReduce,
    BitwiseXorReduce,
    CopysignReduce,
    DivideReduce,
    DivmodReduce,
    EqualReduce,
    FloorDivideReduce,
    FloatPowerReduce,
    FmaxReduce,
    FminReduce,
    FmodReduce,
    GcdReduce,
    GreaterReduce,
    GreaterEqualReduce,
    HeavisideReduce,
    HypotReduce,
    LcmReduce,
    LdexpReduce,
    LeftShiftReduce,
    LessReduce,
    LessEqualReduce,
    LogaddexpReduce,
    Logaddexp2Reduce,
    LogicalAndReduce,
    LogicalOrReduce,
    LogicalXorReduce,
    MaximumReduce,
    MinimumReduce,
    ModReduce,
    MultiplyReduce,
    NextafterReduce,
    NotEqualReduce,
    PowerReduce,
    RemainderReduce,
    RightShiftReduce,
    SubtractReduce,
    TrueDivideReduce,
)


__all__ = [
    "Unary",
    "Binary",
    "ColsBinary",
    "Reduce",
    "func2class_name",
    "unary_module",
    "make_unary",
    "binary_module",
    "make_binary",
    "reduce_module",
    "make_reduce",
    "binary_dict_int_tst",
    "unary_dict_gen_tst",
    "binary_dict_gen_tst",
    "LinearMap",
    "NumExprABC",
    "make_local",
    "make_local_dict",
    "get_ufunc_args",
    "MixUfuncABC",
    "BitwiseNot",
    "Absolute",
    "Arccos",
    "Arccosh",
    "Arcsin",
    "Arcsinh",
    "Arctan",
    "Arctanh",
    "Cbrt",
    "Ceil",
    "Conj",
    "Conjugate",
    "Cos",
    "Cosh",
    "Deg2rad",
    "Degrees",
    "Exp",
    "Exp2",
    "Expm1",
    "Fabs",
    "Floor",
    "Frexp",
    "Invert",
    "Isfinite",
    "Isinf",
    "Isnan",
    "Isnat",
    "Log",
    "Log10",
    "Log1p",
    "Log2",
    "LogicalNot",
    "Modf",
    "Negative",
    "Positive",
    "Rad2deg",
    "Radians",
    "Reciprocal",
    "Rint",
    "Sign",
    "Signbit",
    "Sin",
    "Sinh",
    "Spacing",
    "Sqrt",
    "Square",
    "Tan",
    "Tanh",
    "Trunc",
    "Abs",
    "Add",
    "Arctan2",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Copysign",
    "Divide",
    "Divmod",
    "Equal",
    "FloorDivide",
    "FloatPower",
    "Fmax",
    "Fmin",
    "Fmod",
    "Gcd",
    "Greater",
    "GreaterEqual",
    "Heaviside",
    "Hypot",
    "Lcm",
    "Ldexp",
    "LeftShift",
    "Less",
    "LessEqual",
    "Logaddexp",
    "Logaddexp2",
    "LogicalAnd",
    "LogicalOr",
    "LogicalXor",
    "Maximum",
    "Minimum",
    "Mod",
    "Multiply",
    "Nextafter",
    "NotEqual",
    "Power",
    "Remainder",
    "RightShift",
    "Subtract",
    "TrueDivide",
    "ColsAdd",
    "ColsArctan2",
    "ColsBitwiseAnd",
    "ColsBitwiseOr",
    "ColsBitwiseXor",
    "ColsCopysign",
    "ColsDivide",
    "ColsDivmod",
    "ColsEqual",
    "ColsFloorDivide",
    "ColsFloatPower",
    "ColsFmax",
    "ColsFmin",
    "ColsFmod",
    "ColsGcd",
    "ColsGreater",
    "ColsGreaterEqual",
    "ColsHeaviside",
    "ColsHypot",
    "ColsLcm",
    "ColsLdexp",
    "ColsLeftShift",
    "ColsLess",
    "ColsLessEqual",
    "ColsLogaddexp",
    "ColsLogaddexp2",
    "ColsLogicalAnd",
    "ColsLogicalOr",
    "ColsLogicalXor",
    "ColsMaximum",
    "ColsMinimum",
    "ColsMod",
    "ColsMultiply",
    "ColsNextafter",
    "ColsNotEqual",
    "ColsPower",
    "ColsRemainder",
    "ColsRightShift",
    "ColsSubtract",
    "ColsTrueDivide",
    "AddReduce",
    "Arctan2Reduce",
    "BitwiseAndReduce",
    "BitwiseOrReduce",
    "BitwiseXorReduce",
    "CopysignReduce",
    "DivideReduce",
    "DivmodReduce",
    "EqualReduce",
    "FloorDivideReduce",
    "FloatPowerReduce",
    "FmaxReduce",
    "FminReduce",
    "FmodReduce",
    "GcdReduce",
    "GreaterReduce",
    "GreaterEqualReduce",
    "HeavisideReduce",
    "HypotReduce",
    "LcmReduce",
    "LdexpReduce",
    "LeftShiftReduce",
    "LessReduce",
    "LessEqualReduce",
    "LogaddexpReduce",
    "Logaddexp2Reduce",
    "LogicalAndReduce",
    "LogicalOrReduce",
    "LogicalXorReduce",
    "MaximumReduce",
    "MinimumReduce",
    "ModReduce",
    "MultiplyReduce",
    "NextafterReduce",
    "NotEqualReduce",
    "PowerReduce",
    "RemainderReduce",
    "RightShiftReduce",
    "SubtractReduce",
    "TrueDivideReduce",
]
