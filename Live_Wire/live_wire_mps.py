from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------- Symbolic Boolean expression DSL ----------

class Expr:  # bool output for gates
    """Base class for symbolic Boolean circuit expressions."""

    def __and__(self, other: object) -> "Expr":
        return Gate("AND", (to_expr(self), to_expr(other)))

    def __rand__(self, other: object) -> "Expr":
        return Gate("AND", (to_expr(other), to_expr(self)))

    def __or__(self, other: object) -> "Expr":
        return Gate("OR", (to_expr(self), to_expr(other)))

    def __ror__(self, other: object) -> "Expr":
        return Gate("OR", (to_expr(other), to_expr(self)))

    def __xor__(self, other: object) -> "Expr":
        return Gate("XOR", (to_expr(self), to_expr(other)))

    def __rxor__(self, other: object) -> "Expr":
        return Gate("XOR", (to_expr(other), to_expr(self)))

    def __invert__(self) -> "Expr":
        return Gate("NOT", (to_expr(self),))


@dataclass(frozen=True)
class Var(Expr):
    """Input bit b_idx, with idx starting at 1."""
    idx: int

    def __post_init__(self) -> None:
        if self.idx < 1:
            raise ValueError("Var indices are 1-based and must be >= 1.")

    def __repr__(self) -> str:
        return f"b{self.idx}"


@dataclass(frozen=True)
class Const(Expr):
    """Boolean constant 0 or 1."""
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", int(bool(self.val)))

    def __repr__(self) -> str:
        return str(self.val)


@dataclass(frozen=True)
class Gate(Expr):  # gate object
    """Gate expression. Supported ops: NOT, AND, OR, XOR, NAND, NOR, XNOR."""
    op: str
    args: Tuple[Expr, ...]

    def __post_init__(self) -> None:
        op = self.op.upper()
        object.__setattr__(self, "op", op)

        arity = {
            "NOT": 1,
            "AND": 2,
            "OR": 2,
            "XOR": 2,
            "NAND": 2,
            "NOR": 2,
            "XNOR": 2,
        }.get(op)

        if arity is None:
            raise ValueError(f"Unknown gate op {self.op!r}.")
        if len(self.args) != arity:
            raise ValueError(f"Gate {op} expects {arity} args, got {len(self.args)}.")

    def __repr__(self) -> str:
        if self.op == "NOT":
            return f"(~{self.args[0]!r})"
        sep = {
            "AND": " & ",
            "OR": " | ",
            "XOR": " ^ ",
            "NAND": " NAND ",
            "NOR": " NOR ",
            "XNOR": " XNOR ",
        }[self.op]
        return f"({self.args[0]!r}{sep}{self.args[1]!r})"


def to_expr(x: object) -> Expr:  # converts any object to bool
    if isinstance(x, Expr):
        return x
    if isinstance(x, bool):
        return Const(int(x))
    if isinstance(x, int) and x in (0, 1):
        return Const(x)
    raise TypeError(f"Cannot convert {x!r} to Expr. Use Expr, bool, or int 0/1.")


def bits(N: int) -> Tuple[Var, ...]:  # outputs (b1, b2, b3, b4, b5, b6, b7, b8, b9, b10) for N=10
    """Return (b1, ..., bN)."""
    if N < 0:
        raise ValueError("N must be nonnegative.")
    return tuple(Var(i) for i in range(1, N + 1))


def NOT(x: object) -> Expr:
    return ~to_expr(x)

def AND(a: object, b: object) -> Expr:
    return to_expr(a) & to_expr(b)

def OR(a: object, b: object) -> Expr:
    return to_expr(a) | to_expr(b)

def XOR(a: object, b: object) -> Expr:
    return to_expr(a) ^ to_expr(b)

def NAND(a: object, b: object) -> Expr:
    return Gate("NAND", (to_expr(a), to_expr(b)))

def NOR(a: object, b: object) -> Expr:
    return Gate("NOR", (to_expr(a), to_expr(b)))

def XNOR(a: object, b: object) -> Expr:
    return Gate("XNOR", (to_expr(a), to_expr(b)))

# ---------- Basic expression utilities ----------

def gate_eval(op: str, vals: Sequence[int]) -> int:  # evaluate single gate
    vals = [int(bool(v)) for v in vals]
    if op == "NOT":
        return 1 - vals[0]
    if op == "AND":
        return vals[0] & vals[1]
    if op == "OR":
        return vals[0] | vals[1]
    if op == "XOR":
        return vals[0] ^ vals[1]
    if op == "NAND":
        return 1 - (vals[0] & vals[1])
    if op == "NOR":
        return 1 - (vals[0] | vals[1])
    if op == "XNOR":
        return 1 - (vals[0] ^ vals[1])
    raise ValueError(f"Unknown op {op!r}.")

def eval_expr(expr: Expr, input_bits: Sequence[int]) -> int:  # evaluate bit inputs through expression
    """Evaluate the symbolic expression on a bitstring. input_bits[0] is b1."""
    expr = to_expr(expr)
    memo: Dict[Expr, int] = {}

    def rec(e: Expr) -> int:
        if e in memo:
            return memo[e]
        if isinstance(e, Const):
            out = e.val
        elif isinstance(e, Var):
            if e.idx > len(input_bits):
                raise ValueError(f"Need at least {e.idx} input bits.")
            out = int(bool(input_bits[e.idx - 1]))
        elif isinstance(e, Gate):
            out = gate_eval(e.op, [rec(a) for a in e.args])
        else:
            raise TypeError(e)
        memo[e] = out
        return out

    return rec(expr)

def topo_nodes(expr: Expr) -> List[Expr]:
    """Return all non-constant nodes in dependency-before-use topological order."""
    expr = to_expr(expr)
    seen = set()
    out: List[Expr] = []

    def rec(e: Expr) -> None:
        if isinstance(e, Const):
            return
        if e in seen:
            return
        if isinstance(e, Gate):
            for a in e.args:
                rec(a)
        elif not isinstance(e, Var):
            raise TypeError(e)
        seen.add(e)
        out.append(e)

    rec(expr)
    return out

def max_var_index(expr: Expr) -> int:
    m = 0
    for node in topo_nodes(expr):
        if isinstance(node, Var):
            m = max(m, node.idx)
    return m


# ---------- Live-wire automaton / MPS ----------

@dataclass
class LiveWireMPS:
    """
    Exact deterministic MPS for a Boolean circuit.

    cores_sparse[n][(left_state_index, physical_bit, right_state_index)] = 1
    for n = 0,...,N-1. All omitted entries are 0.

    The dense core convention, from to_dense_cores(), is
        G[n].shape == (chi_n, 2, chi_{n+1})
    and
        F(b1,...,bN) = L @ G[0][:,b1,:] @ ... @ G[N-1][:,bN,:] @ R
    with L = [1,0,0,...]. There is exactly one reachable initial state.
    """
    N: int
    expr: Expr
    live_sets: List[Tuple[Expr, ...]]
    states: List[List[Tuple[int, ...]]]
    cores_sparse: List[Dict[Tuple[int, int, int], int]]
    right_boundary: List[int]
    tau: Dict[Expr, int]
    death: Dict[Expr, int]

    @property
    def bond_dimensions(self) -> List[int]:
        return [len(layer) for layer in self.states]

    def evaluate_bits(self, input_bits: Sequence[int]) -> int:
        """Evaluate by following the deterministic sparse MPS transitions."""
        if len(input_bits) != self.N:
            raise ValueError(f"Expected {self.N} bits, got {len(input_bits)}.")
        state = 0
        for n, bit in enumerate(input_bits):
            b = int(bool(bit))
            core = self.cores_sparse[n]
            matches = [j for (i, bb, j), val in core.items()
                       if val and i == state and bb == b]
            if len(matches) != 1:
                raise RuntimeError(f"Bad transition at site {n + 1}: found {len(matches)} matches.")
            state = matches[0]
        return self.right_boundary[state]

    def contract_dense(self, input_bits: Sequence[int]) -> int:
        """Evaluate by literal dense MPS contraction. Requires numpy."""

        if len(input_bits) != self.N:
            raise ValueError(f"Expected {self.N} bits, got {len(input_bits)}.")

        v = np.zeros(len(self.states[0]), dtype=int)
        v[0] = 1
        for G, bit in zip(self.to_dense_cores(dtype=int), input_bits):
            v = v @ G[:, int(bool(bit)), :]
        return int(v @ np.asarray(self.right_boundary, dtype=int))

    def to_dense_cores(self, dtype=int):
        """Return dense numpy MPS cores with shape (chi_left, 2, chi_right)."""

        dense = []
        for n, core in enumerate(self.cores_sparse):
            G = np.zeros((len(self.states[n]), 2, len(self.states[n + 1])), dtype=dtype)
            for (i, b, j), val in core.items():
                G[i, b, j] = val
            dense.append(G)
        return dense

    def state_dict(self, n: int, state_index: int) -> Dict[str, int]:
        """Human-readable live-wire assignment for a state at cut n."""
        return {repr(w): val for w, val in zip(self.live_sets[n], self.states[n][state_index])}

    def print_summary(self) -> None:
        print(f"N = {self.N}")
        print(f"expr = {self.expr!r}")
        print(f"bond dimensions = {self.bond_dimensions}")
        for n, S in enumerate(self.live_sets):
            print(f"S_{n} = {[repr(w) for w in S]}")


def build_live_wire_mps(expr: Expr, N: Optional[int] = None) -> LiveWireMPS:
    """
    Construct the exact live-wire MPS for expr over bits b1,...,bN.

    If N is omitted, it is inferred as the largest variable index appearing in expr.
    Passing N explicitly is useful when some input bits are unused.

    Example;
        mps = build_live_wire_mps(expr_xor(bits(N)), N=N)
    """
    expr = to_expr(expr)
    inferred_N = max_var_index(expr)
    if N is None:
        N = inferred_N
    if N < inferred_N:
        raise ValueError("N is smaller than the largest variable index in expr.")

    nodes = topo_nodes(expr)
    node_id = {node: i for i, node in enumerate(nodes)}
    input_by_idx = {node.idx: node for node in nodes if isinstance(node, Var)}
    gates = [node for node in nodes if isinstance(node, Gate)]

    # Earliest evaluation time tau.
    tau: Dict[Expr, int] = {}

    def compute_tau(e: Expr) -> int:
        if isinstance(e, Const):
            return 0
        if e in tau:
            return tau[e]
        if isinstance(e, Var):
            t = e.idx
        elif isinstance(e, Gate):
            t = max(compute_tau(a) for a in e.args)
        else:
            raise TypeError(e)
        tau[e] = t
        return t

    compute_tau(expr)

    # Death time: after all gates at time D(w) that need w have been evaluated,
    # w can be dropped. Thus w is live after cut n iff tau(w) <= n < D(w).
    death: Dict[Expr, int] = {node: -1 for node in nodes}
    for g in gates:
        gt = tau[g]
        for a in g.args:
            if not isinstance(a, Const):
                death[a] = max(death[a], gt)

    if not isinstance(expr, Const):
        death[expr] = max(death[expr], N + 1)

    # Gate schedule by time, preserving topological order inside each layer.
    gates_by_time: List[List[Gate]] = [[] for _ in range(N + 1)]
    for g in gates:
        if tau[g] < 0 or tau[g] > N:
            raise RuntimeError(f"Bad tau for gate {g!r}: {tau[g]}")
        gates_by_time[tau[g]].append(g)

    # Live sets after each cut.
    live_sets: List[Tuple[Expr, ...]] = []
    for n in range(N + 1):
        live = [node for node in nodes if tau[node] <= n < death[node]]
        live.sort(key=lambda x: node_id[x])
        live_sets.append(tuple(live))

    def env_value(env: Dict[Expr, int], a: Expr) -> int:
        if isinstance(a, Const):
            return a.val
        if a not in env:
            raise RuntimeError(f"Missing value for {a!r}.")
        return env[a]

    def initial_state() -> Tuple[int, ...]:
        """Evaluate all time-0 gates and form the unique state at cut 0."""
        env: Dict[Expr, int] = {}
        for g in gates_by_time[0]:
            env[g] = gate_eval(g.op, [env_value(env, a) for a in g.args])
        return tuple(env[node] for node in live_sets[0])

    def transition(n: int, prev_state: Tuple[int, ...], input_bit: int) -> Tuple[int, ...]:
        """Transition from cut n-1 to cut n after reading b_n."""
        env: Dict[Expr, int] = {
            node: val for node, val in zip(live_sets[n - 1], prev_state)
        }

        if n in input_by_idx:
            env[input_by_idx[n]] = int(bool(input_bit))

        for g in gates_by_time[n]:
            env[g] = gate_eval(g.op, [env_value(env, a) for a in g.args])

        return tuple(env[node] for node in live_sets[n])

    # Reachable states and sparse transition matrices.
    states: List[List[Tuple[int, ...]]] = [[initial_state()]]
    cores_sparse: List[Dict[Tuple[int, int, int], int]] = []

    for n in range(1, N + 1):
        next_states: List[Tuple[int, ...]] = []
        next_index: Dict[Tuple[int, ...], int] = {}
        core: Dict[Tuple[int, int, int], int] = {}

        for i, state in enumerate(states[n - 1]):
            for b in (0, 1):
                ns = transition(n, state, b)
                if ns not in next_index:
                    next_index[ns] = len(next_states)
                    next_states.append(ns)
                j = next_index[ns]
                core[(i, b, j)] = 1

        states.append(next_states)
        cores_sparse.append(core)

    # Final boundary: read off the output value.
    right_boundary: List[int] = []
    if isinstance(expr, Const):
        right_boundary = [expr.val for _ in states[N]]
    else:
        for s in states[N]:
            env = {node: val for node, val in zip(live_sets[N], s)}
            if expr not in env:
                raise RuntimeError("Output expression is not live at the final cut.")
            right_boundary.append(env[expr])

    return LiveWireMPS(
        N=N,
        expr=expr,
        live_sets=live_sets,
        states=states,
        cores_sparse=cores_sparse,
        right_boundary=right_boundary,
        tau=tau,
        death=death,
    )

def build_mps_from_function(fn: Callable[[Tuple[Var, ...]], Expr], N: int) -> LiveWireMPS:
    """
    Convenience wrapper.

    Example:
        mps = build_mps_from_function(lambda b: (b[0] & b[2]) | ~b[1], N=3)
    """
    return build_live_wire_mps(fn(bits(N)), N=N)

