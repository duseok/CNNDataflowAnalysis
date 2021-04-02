from __future__ import annotations

from sys import maxsize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

    from gekko import GEKKO


class Optimizer:
    def __init__(self, model: GEKKO):
        model.options.SOLVER = 1
        self.__model = model

    def add_constraint(self, eq: List):
        self.__model.Equation(eq)

    def add_constraints(self, eqs: List):
        self.__model.Equations(eqs)

    def minimize(self, eq):
        self.__model.Minimize(eq)

    def maximize(self, eq):
        self.__model.Maximize(eq)

    def solve(self, disp: bool = False):
        model = self.__model
        try:
            model.solve(disp=disp)
            result = model.options.objfcnval
            vars = model._variables
            tf = {str(v.name): v[0] for v in vars}
            return tf, result
        except Exception:
            return None, maxsize

    # mathematic op
    def ceil(self, val):
        model = self.__model
        _val = model.Var(integer=True)
        model.Equations([_val >= val, _val < val + 1])
        return _val
