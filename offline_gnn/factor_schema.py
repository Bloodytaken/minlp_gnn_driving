"""Authored nonlinear-factor metadata for GNN graphs.

SCIP deliberately owns the exact nonlinear constraints.  This module records a
second, read-only description of their residuals so feature extraction never
has to reverse engineer SCIP expression trees.  A factor spec is a mapping
with ``constant + sum(terms)`` on the constrained side; terms are linear,
square differences, bilinear products, or exponentials of a negative square.
"""

from __future__ import annotations

from typing import Any


def is_const(value: Any) -> bool:
    return isinstance(value, (int, float))


def var_name(value: Any) -> str:
    return value if isinstance(value, str) else value.name


def factor(kind: str, sense: str, *, constant: float = 0.0,
           output: Any | None = None) -> dict:
    return {"kind": str(kind), "sense": str(sense),
            "constant": float(constant), "terms": [],
            "outputs": ([] if output is None else [var_name(output)])}


def add_linear(spec: dict, value: Any, coef: float = 1.0,
               *, role: str = "input") -> None:
    if is_const(value):
        spec["constant"] += float(coef) * float(value)
    else:
        spec["terms"].append({"op": "linear", "vars": [var_name(value)],
                              "coef": float(coef), "roles": [str(role)]})


def add_product(spec: dict, left: Any, right: Any, coef: float = 1.0,
                *, roles=("input_left", "input_right")) -> None:
    if is_const(left):
        add_linear(spec, right, float(coef) * float(left), role=roles[1])
    elif is_const(right):
        add_linear(spec, left, float(coef) * float(right), role=roles[0])
    else:
        spec["terms"].append({"op": "bilinear",
                              "vars": [var_name(left), var_name(right)],
                              "coef": float(coef), "roles": list(roles)})


def add_square(spec: dict, value: Any, coef: float = 1.0,
               *, offset: float = 0.0, role: str = "square_input") -> None:
    if is_const(value):
        spec["constant"] += float(coef) * (float(value) - float(offset)) ** 2
    else:
        spec["terms"].append({"op": "square", "vars": [var_name(value)],
                              "coef": float(coef), "offset": float(offset),
                              "roles": [str(role)]})


def add_square_diff(spec: dict, left: Any, right: Any, coef: float = 1.0,
                    *, roles=("diff_left", "diff_right")) -> None:
    if is_const(left) and is_const(right):
        spec["constant"] += float(coef) * (float(left) - float(right)) ** 2
    elif is_const(right):
        add_square(spec, left, coef, offset=float(right), role=roles[0])
    elif is_const(left):
        add_square(spec, right, coef, offset=float(left), role=roles[1])
    else:
        spec["terms"].append({"op": "square_diff",
                              "vars": [var_name(left), var_name(right)],
                              "coef": float(coef), "roles": list(roles)})


def add_exp_neg_square(spec: dict, value: Any, scale: float,
                       coef: float = 1.0, *, role: str = "exp_input") -> None:
    if is_const(value):
        import math
        spec["constant"] += float(coef) * math.exp(float(scale) * float(value) ** 2)
    else:
        spec["terms"].append({"op": "exp_neg_square",
                              "vars": [var_name(value)], "coef": float(coef),
                              "scale": float(scale), "roles": [str(role)]})
