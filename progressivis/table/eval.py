# Borrowed from bcolz.chunked_eval
# Should be adapted to return either a value in a column, or a boolean in a bitmap.
# Should also provide an eval/select progressive module.

import numpy as np

from .column import BaseColumn


def _getvars(expression, user_dict):
    """Get the variables in `expression`."""

    cexpr = compile(expression, '<string>', 'eval')
    exprvars = [var for var in cexpr.co_names
                if var not in ['None', 'False', 'True']]
    reqvars = {}
    for var in exprvars:
        # Get the value
        if var in user_dict:
            val = user_dict[var]
        else:
            val = None
        # Check the value.
        if val is not None:
            reqvars[var] = val
    return reqvars
    
def is_sequence_like(var):
    "Check whether `var` looks like a sequence (strings are not included)."
    if hasattr(var, "__len__"):
        if isinstance(var, (bytes, str)):
            return False
        else:
            return True
    return False

def _eval(expression, user_dict=None, blen=None, **kwargs):
    variables = _getvars(expression, user_dict)
    typesize, vlen = 0, 1
    for name in variables:
        var = variables[name]
        if is_sequence_like(var) and not hasattr(var, "dtype"):
            raise ValueError("only numpy/column sequences supported")
        if hasattr(var, "dtype") and not hasattr(var, "__len__"):
            continue
        if hasattr(var, "dtype"):  # numpy/carray arrays
            if isinstance(var, np.ndarray):  # numpy array
                typesize += var.dtype.itemsize * np.prod(var.shape[1:])
            elif isinstance(var, BaseColumn): 
                typesize += var.dtype.itemsize
            else:
                raise ValueError("only numpy/Column objects supported")
        if is_sequence_like(var):
            if vlen > 1 and vlen != len(var):
                raise ValueError("arrays must have the same length")
            vlen = len(var)

    if typesize == 0:
        # All scalars
        #pylint: disable=eval-used
        return eval(expression, variables)
    return _eval_blocks(expression, variables, vlen, typesize, blen, **kwargs)
    
def _eval_blocks(expression, variables, vlen, typesize, blen, **kwargs):
    """Perform the evaluation in blocks."""

    if not blen:
        # Compute the optimal block size (in elements)
        # The next is based on nothing so far, but should be based on experiments.
        bsize = 2**16
        blen = int(bsize / typesize)
        # Protection against too large atomsizes
        if blen == 0:
            blen = 1

    vars_ = {}
    # Get containers for vars
    maxndims = 0
    for name in variables:
        var = variables[name]
        if is_sequence_like(var):
            ndims = len(var.shape) + len(var.dtype.shape)
            if ndims > maxndims:
                maxndims = ndims
            if len(var) > blen and hasattr(var, "_getrange"):
                shape = (blen, ) + var.shape[1:]
                vars_[name] = np.empty(shape, dtype=var.dtype)

    for i in range(0, vlen, blen):
        # Fill buffers for vars
        for name in variables:
            var = variables[name]
            if is_sequence_like(var) and len(var) > blen:
                vars_[name] = var[i:i+blen]
            else:
                if hasattr(var, "__getitem__"):
                    vars_[name] = var[:]
                else:
                    vars_[name] = var

        #pylint: disable=eval-used
        res_block = eval(expression, vars_)

        if i == 0:
            # Detection of reduction operations
            scalar = False
            dim_reduction = False
            if len(res_block.shape) == 0:
                scalar = True
                result = res_block
                continue
            elif len(res_block.shape) < maxndims:
                dim_reduction = True
                result = res_block
                continue
            out_shape = list(res_block.shape)
            out_shape[0] = vlen
            result = np.empty(out_shape, dtype=res_block.dtype)
            result[:blen] = res_block
        else:
            if scalar or dim_reduction:
                result += res_block
            result[i:i+blen] = res_block

    if scalar:
        return result[()]
    return result
