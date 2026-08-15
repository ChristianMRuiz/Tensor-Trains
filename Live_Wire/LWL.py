"""
Complex N=8 tests for live_wire_mps.py.

Put this file next to live_wire_mps.py and run either

    python test_live_wire_mps_n8.py

or, if you use pytest,

    pytest -q test_live_wire_mps_n8.py

Each case is exhaustively checked over all 2^8 bitstrings.  The reference
answers are computed by independent pure-Python functions where possible,
rather than by live_wire_mps.eval_expr.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

from .live_wire_mps import (
    NOT,
    AND,
    NAND,
    NOR,
    XNOR,
    Const,
    Expr,
    build_live_wire_mps,
    bits,
    Var,
)

BitTuple = Tuple[int, ...]
RefFn = Callable[[BitTuple], int]


def as_bit(x: object) -> int:
    return int(bool(x))

def expr_or(xs: Iterable[Expr]) -> Expr:
    xs = list(xs)
    if not xs:
        return Const(0)
    out = xs[0]
    for x in xs[1:]:
        out = out | x
    return out

def expr_and(xs: Iterable[Expr]) -> Expr:
    xs = list(xs)
    if not xs:
        return Const(1)
    out = xs[0]
    for x in xs[1:]:
        out = out & x
    return out

def expr_xor(xs: Iterable[Expr]) -> Expr:
    xs = list(xs)
    if not xs:
        return Const(0)
    out = xs[0]
    for x in xs[1:]:
        out = out ^ x
    return out

def ite(c: Expr, t: Expr, f: Expr) -> Expr:
    """If c then t else f, as a Boolean expression."""
    return (c & t) | (~c & f)

def count_at_least_expr(xs: Sequence[Expr], k: int) -> Expr:  # checks number of 1's greater
    """
    Boolean expression for sum(xs) >= k.

    Dynamic-programming circuit.  state[j] means that the number of ones seen
    so far is exactly j for j < k, and at least k for j == k.
    """
    if k <= 0:
        return Const(1)
    state: List[Expr] = [Const(1)] + [Const(0) for _ in range(k)]
    for x in xs:
        new: List[Expr] = [Const(0) for _ in range(k + 1)]
        new[0] = state[0] & ~x
        for j in range(1, k):
            new[j] = (state[j] & ~x) | (state[j - 1] & x)
        new[k] = state[k] | (state[k - 1] & x)
        state = new
    return state[k]

def count_mod_expr(xs: Sequence[Expr], modulus: int, target: int) -> Expr:  # checks number of 1's equal
    """Boolean expression for sum(xs) == target mod modulus."""
    state: List[Expr] = [Const(0) for _ in range(modulus)]
    state[0] = Const(1)
    for x in xs:
        new: List[Expr] = [Const(0) for _ in range(modulus)]
        for r in range(modulus):
            new[r] = new[r] | (state[r] & ~x)
            new[(r + 1) % modulus] = new[(r + 1) % modulus] | (state[r] & x)
        state = new
    return state[target % modulus]

def ge_constant_msb_expr(xs: Sequence[Expr], const_bits: Sequence[int]) -> Expr:  # greater then input bits
    """
    Boolean expression for the MSB-first bitstring xs being >= const_bits.

    Example: xs=(b1,...,b8), const_bits=(1,0,1,0,1,1,0,1) tests
    int(b1...b8, base=2) >= 0b10101101.
    """
    if len(xs) != len(const_bits):
        raise ValueError("xs and const_bits must have the same length")
    gt: Expr = Const(0)
    eq: Expr = Const(1)
    for x, c in zip(xs, const_bits):
        c = int(bool(c))
        if c == 0:
            gt = gt | (eq & x)
            eq = eq & ~x
        else:
            eq = eq & x
    return gt | eq

def unsigned_ge_expr(a: Sequence[Expr], b: Sequence[Expr]) -> Expr:  # ???
    """Expression for unsigned MSB-first integer a >= b."""
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    gt: Expr = Const(0)
    eq: Expr = Const(1)
    for ai, bi in zip(a, b):
        gt = gt | (eq & ai & ~bi)
        eq = eq & XNOR(ai, bi)
    return gt | eq

def minterm_expr(xs: Sequence[Expr], target: Sequence[int]) -> Expr:
    """Conjunction that is true exactly on one bitstring."""
    return expr_and(x if bit else ~x for x, bit in zip(xs, target))

def dnf_from_reference(xs: Sequence[Expr], ref: RefFn) -> Expr:
    """DNF expression for an arbitrary N-bit reference function."""
    terms = []
    for bitstring in product((0, 1), repeat=len(xs)):
        if ref(tuple(bitstring)):
            terms.append(minterm_expr(xs, bitstring))
    return expr_or(terms)


# ---------- My added functions ----------

def LW_Rect(Stop, N, inv=False):
    """
    Assumes that the function wants a number and gives back zero or one
    """

    stop_bits = tuple(int(bit_) for bit_ in Stop)

    if inv:
        expression = ge_constant_msb_expr(bits(N), stop_bits)
    else:
        expression = ~ge_constant_msb_expr(bits(N), stop_bits)

    mps = build_live_wire_mps(expression, N)
    cores = mps.to_dense_cores(dtype=float)
    R = np.array(mps.right_boundary, dtype=float)
    # to_dense_cores() omits right_boundary; absorb it into the last core so
    # TTL's Train_to_Func (which uses np.trace) gives the correct output.
    #cores.append(R)
    cores[-1] = np.einsum('ijk,k->ij', cores[-1], R)[:, :, np.newaxis]
    return cores

def LW_iSquare(Val, N):
    bits = [0] + [1]*(N-Val)
    left_b = np.append(bits, [0]*(Val-1))
    bits[-1] = 1-bits[-1]  # shifting due to 'greater then' vs 'greater then or equal'
    right_b = np.append(1-np.array(bits), [0]*(Val-1))

    one = LW_Rect(left_b, N)
    two = LW_Rect(right_b, N, inv=True)

    return one, two

def Func_to_Train_LW(f, N):
    """
        Assumes that the function given takes a list of bits.
        (In reality the same convinience wrapper made by him)
    """
    return build_live_wire_mps(f(bits(N)), N=N)

