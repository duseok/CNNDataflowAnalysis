from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING

from Dataflow.dimension import Dimension

if TYPE_CHECKING:
    from Dataflow.tiling_factor import TilingFactor
    from param import Param


class DataType(Enum):
    Input = auto()
    Output = auto()
    Weight = auto()


class Data(ABC):
    dependency = None

    def __init__(self, size: int, param: Param):
        self.type: DataType
        self.size = self.total_size = size
        self._param = param

    def set_tiling_factor(self, tf: TilingFactor):
        self._tiling_factor = tf
        self._set_tiled_data_size(self._param)

    @abstractmethod
    def _set_tiled_data_size(self, param: Param):
        raise NotImplementedError


class InputData(Data):
    dependency = [Dimension.Batch, Dimension.InChan, Dimension.Height, Dimension.Width]

    def __init__(self, size: int, param: Param):
        super().__init__(size, param)
        self.type = DataType.Input

    def _set_tiled_data_size(self, param: Param):
        tf = self._tiling_factor
        s = param.op.stride
        k = param.op.kernel
        self.size = tf.C * (s * tf.H + k - s) * (s * tf.W + k - s)


class OutputData(Data):
    dependency = [
        Dimension.Batch,
        Dimension.OutChan,
        Dimension.Height,
        Dimension.Width,
    ]

    def __init__(self, size: int, param: Param):
        super().__init__(size, param)
        self.type = DataType.Output

    def _set_tiled_data_size(self, param: Param):
        tf = self._tiling_factor
        self.size = tf.N * tf.H * tf.W


class ConvWeightData(Data):
    dependency = [Dimension.OutChan, Dimension.InChan]

    def __init__(self, size: int, param: Param):
        super().__init__(size, param)
        self.type = DataType.Weight

    def _set_tiled_data_size(self, param: Param):
        tf = self._tiling_factor
        k = param.op.kernel
        self.size = tf.N * tf.C * k ** 2


class DWConvWeightData(Data):
    pass
