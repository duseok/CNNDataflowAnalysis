from __future__ import annotations

from typing import TYPE_CHECKING

from Dataflow.dimension import Dimension
from Dataflow.tiling_factor import TilingFactor

if TYPE_CHECKING:
    from typing import List

    from Dataflow.mem_op import MemOp


class Loop:
    def __init__(self, dim: Dimension, size: int, ops: List[MemOp] = []):
        self.dim = dim
        self.size = size
        self.mem_ops = ops.copy()

    def set_tiling_factor(self, tf: TilingFactor):
        self.__calc_repeat_count(self.dim, tf)

    def __calc_repeat_count(self, dim: str, tf: TilingFactor):
        size = self.size
        if dim == Dimension.Batch:
            self.repeat_count = size
        elif dim == Dimension.InChan:
            self.repeat_count = size / tf.C
        elif dim == Dimension.OutChan:
            self.repeat_count = size / tf.N
        elif dim == Dimension.Height:
            self.repeat_count = size / tf.H
        elif dim == Dimension.Width:
            self.repeat_count = size / tf.W

    def __repr__(self) -> str:
        return f"{self.dim.name:7s}\t{self.size}\t{[op for op in self.mem_ops]}"
