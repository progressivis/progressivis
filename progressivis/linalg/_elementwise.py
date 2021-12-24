from __future__ import annotations

from .elementwise import Unary, Binary, ColsBinary, Reduce

import numpy as np


class BitwiseNot(Unary):
    def __init__(self, *args, **kwds):
        super(BitwiseNot, self).__init__(np.bitwise_not, **kwds)


class Absolute(Unary):
    def __init__(self, *args, **kwds):
        super(Absolute, self).__init__(np.absolute, **kwds)


class Arccos(Unary):
    def __init__(self, *args, **kwds):
        super(Arccos, self).__init__(np.arccos, **kwds)


class Arccosh(Unary):
    def __init__(self, *args, **kwds):
        super(Arccosh, self).__init__(np.arccosh, **kwds)


class Arcsin(Unary):
    def __init__(self, *args, **kwds):
        super(Arcsin, self).__init__(np.arcsin, **kwds)


class Arcsinh(Unary):
    def __init__(self, *args, **kwds):
        super(Arcsinh, self).__init__(np.arcsinh, **kwds)


class Arctan(Unary):
    def __init__(self, *args, **kwds):
        super(Arctan, self).__init__(np.arctan, **kwds)


class Arctanh(Unary):
    def __init__(self, *args, **kwds):
        super(Arctanh, self).__init__(np.arctanh, **kwds)


class Cbrt(Unary):
    def __init__(self, *args, **kwds):
        super(Cbrt, self).__init__(np.cbrt, **kwds)


class Ceil(Unary):
    def __init__(self, *args, **kwds):
        super(Ceil, self).__init__(np.ceil, **kwds)


class Conj(Unary):
    def __init__(self, *args, **kwds):
        super(Conj, self).__init__(np.conj, **kwds)


class Conjugate(Unary):
    def __init__(self, *args, **kwds):
        super(Conjugate, self).__init__(np.conjugate, **kwds)


class Cos(Unary):
    def __init__(self, *args, **kwds):
        super(Cos, self).__init__(np.cos, **kwds)


class Cosh(Unary):
    def __init__(self, *args, **kwds):
        super(Cosh, self).__init__(np.cosh, **kwds)


class Deg2rad(Unary):
    def __init__(self, *args, **kwds):
        super(Deg2rad, self).__init__(np.deg2rad, **kwds)


class Degrees(Unary):
    def __init__(self, *args, **kwds):
        super(Degrees, self).__init__(np.degrees, **kwds)


class Exp(Unary):
    def __init__(self, *args, **kwds):
        super(Exp, self).__init__(np.exp, **kwds)


class Exp2(Unary):
    def __init__(self, *args, **kwds):
        super(Exp2, self).__init__(np.exp2, **kwds)


class Expm1(Unary):
    def __init__(self, *args, **kwds):
        super(Expm1, self).__init__(np.expm1, **kwds)


class Fabs(Unary):
    def __init__(self, *args, **kwds):
        super(Fabs, self).__init__(np.fabs, **kwds)


class Floor(Unary):
    def __init__(self, *args, **kwds):
        super(Floor, self).__init__(np.floor, **kwds)


class Frexp(Unary):
    def __init__(self, *args, **kwds):
        super(Frexp, self).__init__(np.frexp, **kwds)


class Invert(Unary):
    def __init__(self, *args, **kwds):
        super(Invert, self).__init__(np.invert, **kwds)


class Isfinite(Unary):
    def __init__(self, *args, **kwds):
        super(Isfinite, self).__init__(np.isfinite, **kwds)


class Isinf(Unary):
    def __init__(self, *args, **kwds):
        super(Isinf, self).__init__(np.isinf, **kwds)


class Isnan(Unary):
    def __init__(self, *args, **kwds):
        super(Isnan, self).__init__(np.isnan, **kwds)


class Isnat(Unary):
    def __init__(self, *args, **kwds):
        super(Isnat, self).__init__(np.isnat, **kwds)


class Log(Unary):
    def __init__(self, *args, **kwds):
        super(Log, self).__init__(np.log, **kwds)


class Log10(Unary):
    def __init__(self, *args, **kwds):
        super(Log10, self).__init__(np.log10, **kwds)


class Log1p(Unary):
    def __init__(self, *args, **kwds):
        super(Log1p, self).__init__(np.log1p, **kwds)


class Log2(Unary):
    def __init__(self, *args, **kwds):
        super(Log2, self).__init__(np.log2, **kwds)


class LogicalNot(Unary):
    def __init__(self, *args, **kwds):
        super(LogicalNot, self).__init__(np.logical_not, **kwds)


class Modf(Unary):
    def __init__(self, *args, **kwds):
        super(Modf, self).__init__(np.modf, **kwds)


class Negative(Unary):
    def __init__(self, *args, **kwds):
        super(Negative, self).__init__(np.negative, **kwds)


class Positive(Unary):
    def __init__(self, *args, **kwds):
        super(Positive, self).__init__(np.positive, **kwds)


class Rad2deg(Unary):
    def __init__(self, *args, **kwds):
        super(Rad2deg, self).__init__(np.rad2deg, **kwds)


class Radians(Unary):
    def __init__(self, *args, **kwds):
        super(Radians, self).__init__(np.radians, **kwds)


class Reciprocal(Unary):
    def __init__(self, *args, **kwds):
        super(Reciprocal, self).__init__(np.reciprocal, **kwds)


class Rint(Unary):
    def __init__(self, *args, **kwds):
        super(Rint, self).__init__(np.rint, **kwds)


class Sign(Unary):
    def __init__(self, *args, **kwds):
        super(Sign, self).__init__(np.sign, **kwds)


class Signbit(Unary):
    def __init__(self, *args, **kwds):
        super(Signbit, self).__init__(np.signbit, **kwds)


class Sin(Unary):
    def __init__(self, *args, **kwds):
        super(Sin, self).__init__(np.sin, **kwds)


class Sinh(Unary):
    def __init__(self, *args, **kwds):
        super(Sinh, self).__init__(np.sinh, **kwds)


class Spacing(Unary):
    def __init__(self, *args, **kwds):
        super(Spacing, self).__init__(np.spacing, **kwds)


class Sqrt(Unary):
    def __init__(self, *args, **kwds):
        super(Sqrt, self).__init__(np.sqrt, **kwds)


class Square(Unary):
    def __init__(self, *args, **kwds):
        super(Square, self).__init__(np.square, **kwds)


class Tan(Unary):
    def __init__(self, *args, **kwds):
        super(Tan, self).__init__(np.tan, **kwds)


class Tanh(Unary):
    def __init__(self, *args, **kwds):
        super(Tanh, self).__init__(np.tanh, **kwds)


class Trunc(Unary):
    def __init__(self, *args, **kwds):
        super(Trunc, self).__init__(np.trunc, **kwds)


class Abs(Unary):
    def __init__(self, *args, **kwds):
        super(Abs, self).__init__(np.abs, **kwds)


class Add(Binary):
    def __init__(self, *args, **kwds):
        super(Add, self).__init__(np.add, **kwds)


class Arctan2(Binary):
    def __init__(self, *args, **kwds):
        super(Arctan2, self).__init__(np.arctan2, **kwds)


class BitwiseAnd(Binary):
    def __init__(self, *args, **kwds):
        super(BitwiseAnd, self).__init__(np.bitwise_and, **kwds)


class BitwiseOr(Binary):
    def __init__(self, *args, **kwds):
        super(BitwiseOr, self).__init__(np.bitwise_or, **kwds)


class BitwiseXor(Binary):
    def __init__(self, *args, **kwds):
        super(BitwiseXor, self).__init__(np.bitwise_xor, **kwds)


class Copysign(Binary):
    def __init__(self, *args, **kwds):
        super(Copysign, self).__init__(np.copysign, **kwds)


class Divide(Binary):
    def __init__(self, *args, **kwds):
        super(Divide, self).__init__(np.divide, **kwds)


class Divmod(Binary):
    def __init__(self, *args, **kwds):
        super(Divmod, self).__init__(np.divmod, **kwds)


class Equal(Binary):
    def __init__(self, *args, **kwds):
        super(Equal, self).__init__(np.equal, **kwds)


class FloorDivide(Binary):
    def __init__(self, *args, **kwds):
        super(FloorDivide, self).__init__(np.floor_divide, **kwds)


class FloatPower(Binary):
    def __init__(self, *args, **kwds):
        super(FloatPower, self).__init__(np.float_power, **kwds)


class Fmax(Binary):
    def __init__(self, *args, **kwds):
        super(Fmax, self).__init__(np.fmax, **kwds)


class Fmin(Binary):
    def __init__(self, *args, **kwds):
        super(Fmin, self).__init__(np.fmin, **kwds)


class Fmod(Binary):
    def __init__(self, *args, **kwds):
        super(Fmod, self).__init__(np.fmod, **kwds)


class Gcd(Binary):
    def __init__(self, *args, **kwds):
        super(Gcd, self).__init__(np.gcd, **kwds)


class Greater(Binary):
    def __init__(self, *args, **kwds):
        super(Greater, self).__init__(np.greater, **kwds)


class GreaterEqual(Binary):
    def __init__(self, *args, **kwds):
        super(GreaterEqual, self).__init__(np.greater_equal, **kwds)


class Heaviside(Binary):
    def __init__(self, *args, **kwds):
        super(Heaviside, self).__init__(np.heaviside, **kwds)


class Hypot(Binary):
    def __init__(self, *args, **kwds):
        super(Hypot, self).__init__(np.hypot, **kwds)


class Lcm(Binary):
    def __init__(self, *args, **kwds):
        super(Lcm, self).__init__(np.lcm, **kwds)


class Ldexp(Binary):
    def __init__(self, *args, **kwds):
        super(Ldexp, self).__init__(np.ldexp, **kwds)


class LeftShift(Binary):
    def __init__(self, *args, **kwds):
        super(LeftShift, self).__init__(np.left_shift, **kwds)


class Less(Binary):
    def __init__(self, *args, **kwds):
        super(Less, self).__init__(np.less, **kwds)


class LessEqual(Binary):
    def __init__(self, *args, **kwds):
        super(LessEqual, self).__init__(np.less_equal, **kwds)


class Logaddexp(Binary):
    def __init__(self, *args, **kwds):
        super(Logaddexp, self).__init__(np.logaddexp, **kwds)


class Logaddexp2(Binary):
    def __init__(self, *args, **kwds):
        super(Logaddexp2, self).__init__(np.logaddexp2, **kwds)


class LogicalAnd(Binary):
    def __init__(self, *args, **kwds):
        super(LogicalAnd, self).__init__(np.logical_and, **kwds)


class LogicalOr(Binary):
    def __init__(self, *args, **kwds):
        super(LogicalOr, self).__init__(np.logical_or, **kwds)


class LogicalXor(Binary):
    def __init__(self, *args, **kwds):
        super(LogicalXor, self).__init__(np.logical_xor, **kwds)


class Maximum(Binary):
    def __init__(self, *args, **kwds):
        super(Maximum, self).__init__(np.maximum, **kwds)


class Minimum(Binary):
    def __init__(self, *args, **kwds):
        super(Minimum, self).__init__(np.minimum, **kwds)


class Mod(Binary):
    def __init__(self, *args, **kwds):
        super(Mod, self).__init__(np.mod, **kwds)


class Multiply(Binary):
    def __init__(self, *args, **kwds):
        super(Multiply, self).__init__(np.multiply, **kwds)


class Nextafter(Binary):
    def __init__(self, *args, **kwds):
        super(Nextafter, self).__init__(np.nextafter, **kwds)


class NotEqual(Binary):
    def __init__(self, *args, **kwds):
        super(NotEqual, self).__init__(np.not_equal, **kwds)


class Power(Binary):
    def __init__(self, *args, **kwds):
        super(Power, self).__init__(np.power, **kwds)


class Remainder(Binary):
    def __init__(self, *args, **kwds):
        super(Remainder, self).__init__(np.remainder, **kwds)


class RightShift(Binary):
    def __init__(self, *args, **kwds):
        super(RightShift, self).__init__(np.right_shift, **kwds)


class Subtract(Binary):
    def __init__(self, *args, **kwds):
        super(Subtract, self).__init__(np.subtract, **kwds)


class TrueDivide(Binary):
    def __init__(self, *args, **kwds):
        super(TrueDivide, self).__init__(np.true_divide, **kwds)


class ColsAdd(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsAdd, self).__init__(np.add, **kwds)


class ColsArctan2(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsArctan2, self).__init__(np.arctan2, **kwds)


class ColsBitwiseAnd(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsBitwiseAnd, self).__init__(np.bitwise_and, **kwds)


class ColsBitwiseOr(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsBitwiseOr, self).__init__(np.bitwise_or, **kwds)


class ColsBitwiseXor(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsBitwiseXor, self).__init__(np.bitwise_xor, **kwds)


class ColsCopysign(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsCopysign, self).__init__(np.copysign, **kwds)


class ColsDivide(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsDivide, self).__init__(np.divide, **kwds)


class ColsDivmod(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsDivmod, self).__init__(np.divmod, **kwds)


class ColsEqual(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsEqual, self).__init__(np.equal, **kwds)


class ColsFloorDivide(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsFloorDivide, self).__init__(np.floor_divide, **kwds)


class ColsFloatPower(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsFloatPower, self).__init__(np.float_power, **kwds)


class ColsFmax(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsFmax, self).__init__(np.fmax, **kwds)


class ColsFmin(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsFmin, self).__init__(np.fmin, **kwds)


class ColsFmod(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsFmod, self).__init__(np.fmod, **kwds)


class ColsGcd(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsGcd, self).__init__(np.gcd, **kwds)


class ColsGreater(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsGreater, self).__init__(np.greater, **kwds)


class ColsGreaterEqual(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsGreaterEqual, self).__init__(np.greater_equal, **kwds)


class ColsHeaviside(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsHeaviside, self).__init__(np.heaviside, **kwds)


class ColsHypot(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsHypot, self).__init__(np.hypot, **kwds)


class ColsLcm(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLcm, self).__init__(np.lcm, **kwds)


class ColsLdexp(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLdexp, self).__init__(np.ldexp, **kwds)


class ColsLeftShift(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLeftShift, self).__init__(np.left_shift, **kwds)


class ColsLess(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLess, self).__init__(np.less, **kwds)


class ColsLessEqual(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLessEqual, self).__init__(np.less_equal, **kwds)


class ColsLogaddexp(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLogaddexp, self).__init__(np.logaddexp, **kwds)


class ColsLogaddexp2(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLogaddexp2, self).__init__(np.logaddexp2, **kwds)


class ColsLogicalAnd(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLogicalAnd, self).__init__(np.logical_and, **kwds)


class ColsLogicalOr(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLogicalOr, self).__init__(np.logical_or, **kwds)


class ColsLogicalXor(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsLogicalXor, self).__init__(np.logical_xor, **kwds)


class ColsMaximum(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsMaximum, self).__init__(np.maximum, **kwds)


class ColsMinimum(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsMinimum, self).__init__(np.minimum, **kwds)


class ColsMod(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsMod, self).__init__(np.mod, **kwds)


class ColsMultiply(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsMultiply, self).__init__(np.multiply, **kwds)


class ColsNextafter(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsNextafter, self).__init__(np.nextafter, **kwds)


class ColsNotEqual(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsNotEqual, self).__init__(np.not_equal, **kwds)


class ColsPower(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsPower, self).__init__(np.power, **kwds)


class ColsRemainder(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsRemainder, self).__init__(np.remainder, **kwds)


class ColsRightShift(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsRightShift, self).__init__(np.right_shift, **kwds)


class ColsSubtract(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsSubtract, self).__init__(np.subtract, **kwds)


class ColsTrueDivide(ColsBinary):
    def __init__(self, *args, **kwds):
        super(ColsTrueDivide, self).__init__(np.true_divide, **kwds)


class AddReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(AddReduce, self).__init__(np.add, **kwds)


class Arctan2Reduce(Reduce):
    def __init__(self, *args, **kwds):
        super(Arctan2Reduce, self).__init__(np.arctan2, **kwds)


class BitwiseAndReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(BitwiseAndReduce, self).__init__(np.bitwise_and, **kwds)


class BitwiseOrReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(BitwiseOrReduce, self).__init__(np.bitwise_or, **kwds)


class BitwiseXorReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(BitwiseXorReduce, self).__init__(np.bitwise_xor, **kwds)


class CopysignReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(CopysignReduce, self).__init__(np.copysign, **kwds)


class DivideReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(DivideReduce, self).__init__(np.divide, **kwds)


class DivmodReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(DivmodReduce, self).__init__(np.divmod, **kwds)


class EqualReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(EqualReduce, self).__init__(np.equal, **kwds)


class FloorDivideReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(FloorDivideReduce, self).__init__(np.floor_divide, **kwds)


class FloatPowerReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(FloatPowerReduce, self).__init__(np.float_power, **kwds)


class FmaxReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(FmaxReduce, self).__init__(np.fmax, **kwds)


class FminReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(FminReduce, self).__init__(np.fmin, **kwds)


class FmodReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(FmodReduce, self).__init__(np.fmod, **kwds)


class GcdReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(GcdReduce, self).__init__(np.gcd, **kwds)


class GreaterReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(GreaterReduce, self).__init__(np.greater, **kwds)


class GreaterEqualReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(GreaterEqualReduce, self).__init__(np.greater_equal, **kwds)


class HeavisideReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(HeavisideReduce, self).__init__(np.heaviside, **kwds)


class HypotReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(HypotReduce, self).__init__(np.hypot, **kwds)


class LcmReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(LcmReduce, self).__init__(np.lcm, **kwds)


class LdexpReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(LdexpReduce, self).__init__(np.ldexp, **kwds)


class LeftShiftReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(LeftShiftReduce, self).__init__(np.left_shift, **kwds)


class LessReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(LessReduce, self).__init__(np.less, **kwds)


class LessEqualReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(LessEqualReduce, self).__init__(np.less_equal, **kwds)


class LogaddexpReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(LogaddexpReduce, self).__init__(np.logaddexp, **kwds)


class Logaddexp2Reduce(Reduce):
    def __init__(self, *args, **kwds):
        super(Logaddexp2Reduce, self).__init__(np.logaddexp2, **kwds)


class LogicalAndReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(LogicalAndReduce, self).__init__(np.logical_and, **kwds)


class LogicalOrReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(LogicalOrReduce, self).__init__(np.logical_or, **kwds)


class LogicalXorReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(LogicalXorReduce, self).__init__(np.logical_xor, **kwds)


class MaximumReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(MaximumReduce, self).__init__(np.maximum, **kwds)


class MinimumReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(MinimumReduce, self).__init__(np.minimum, **kwds)


class ModReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(ModReduce, self).__init__(np.mod, **kwds)


class MultiplyReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(MultiplyReduce, self).__init__(np.multiply, **kwds)


class NextafterReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(NextafterReduce, self).__init__(np.nextafter, **kwds)


class NotEqualReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(NotEqualReduce, self).__init__(np.not_equal, **kwds)


class PowerReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(PowerReduce, self).__init__(np.power, **kwds)


class RemainderReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(RemainderReduce, self).__init__(np.remainder, **kwds)


class RightShiftReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(RightShiftReduce, self).__init__(np.right_shift, **kwds)


class SubtractReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(SubtractReduce, self).__init__(np.subtract, **kwds)


class TrueDivideReduce(Reduce):
    def __init__(self, *args, **kwds):
        super(TrueDivideReduce, self).__init__(np.true_divide, **kwds)
