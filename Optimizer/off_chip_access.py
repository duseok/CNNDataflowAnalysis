from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from Dataflow.data import DataType
from Dataflow.tiling_factor import TilingFactor
from Optimizer.optimizer import Optimizer

if TYPE_CHECKING:
    from typing import Dict, Tuple

    from Dataflow.data import Data
    from Dataflow.loop import Loop
    from param import Param


class Objective(ABC):
    def __init__(
        self,
        optimizer: Optimizer,
        param: Param,
        dataflow: Tuple[Loop],
        tf: TilingFactor,
    ):
        self._optimizer = optimizer
        self._param = param
        self._dataflow = dataflow
        self._tf = tf
        self._data = self.__get_data()
        self.__set_tiling_factor()

    def __set_tiling_factor(self):
        for data in self._dataflow:
            data.set_tiling_factor(self._tf)
        for data in self._data:
            data.set_tiling_factor(self._tf)

    def __get_data(self) -> Tuple[Data]:
        data = set()
        for loop in self._dataflow:
            data = data.union(set(o.data for o in loop.mem_ops))
        return tuple(data)

    @property
    def total_data_size(self):
        return sum([d.size for d in self._data])

    @abstractmethod
    def set_search_spase(self):
        raise NotImplementedError


class OffChipAccess(Objective):
    def set_search_spase(self):
        self.__set_constraints()
        self.__set_objective()

    def __set_constraints(self):
        opt = self._optimizer
        param = self._param
        mem_size = param.hw.onchip_mem_size

        tf = self._tf
        opt.add_constraint(self.total_data_size <= mem_size)
        opt.add_constraint(tf.N <= param.op.out_chan)
        opt.add_constraint(tf.C <= param.op.in_chan)
        opt.add_constraint(tf.H <= param.height)
        opt.add_constraint(tf.W <= param.width)

    def __set_objective(self):
        param = self._param
        cr = param.compression_ratio

        access_num = self.__get_access_num()
        volume = [cr[d.type] * d.size * access_num[i] for i, d in enumerate(self._data)]
        self._optimizer.minimize(sum(volume))

    def __get_access_num(self) -> Tuple[int]:
        access_num = {d: 1 for d in DataType}
        flag = {d: True for d in DataType}
        for loop in self._dataflow:
            access_num = self.__update_access_num(access_num, loop, flag)
            self.__check_off_chip_access(loop, flag)
        return tuple(access_num.values())

    @staticmethod
    def __check_off_chip_access(loop: Loop, flag: Dict[DataType, bool]):
        access_types = [op.data.type for op in loop.mem_ops]
        for access_type in access_types:
            flag[access_type] = False

    def __update_access_num(
        self, access_num: Dict[DataType, int], loop: Loop, flag: Dict[DataType, bool]
    ):
        count = loop.repeat_count
        __calc_access_num = (
            lambda d: access_num[d.type] * self.__make_mult_val(count, d, loop)
            if flag[d.type]
            else access_num[d.type]
        )
        access_num = {d.type: __calc_access_num(d) for d in self._data}
        return access_num

    def __make_mult_val(self, val, data: Data, loop: Loop):
        opt = self._optimizer
        if loop.dim not in data.dependency:
            _val = opt.ceil(val)
            return (2 * _val - 1) if data.type == DataType.Output else _val
        return val
