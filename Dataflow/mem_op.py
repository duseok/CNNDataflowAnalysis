from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING
from Dataflow.data import Data

if TYPE_CHECKING:
    from typing import List


class MemOpType(Enum):
    LOAD = auto()
    STORE = auto()


class MemOp:
    def __init__(self, data: Data, op: MemOpType, outer_loops: List):
        self.op = op
        self.data = data
        self.outer_loops = outer_loops.copy()

    def __repr__(self) -> str:
        return f"({self.op.name} {self.data.__class__.__name__} / Outer Loops: {[l.dim.name[0] for l in self.outer_loops]})"
