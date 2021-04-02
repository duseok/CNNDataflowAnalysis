from __future__ import annotations

from abc import ABC, abstractclassmethod
from typing import TYPE_CHECKING

from Dataflow.data import ConvWeightData, DataType, InputData, OutputData
from Dataflow.dimension import Dimension
from Dataflow.loop import Loop
from Dataflow.mem_op import MemOp, MemOpType
from Dataflow.tiling_factor import TilingFactor
from gekko import GEKKO
from param import OPParam, Param

if TYPE_CHECKING:
    from typing import Generator, Tuple


class DataflowGeneratorFactory(ABC):
    @classmethod
    def create_generator(cls, param: Param) -> Generator:
        cls._param = param
        dataflows = cls._create_loop_orders()
        for d in dataflows:
            cls._insert_mem_ops(d)
            yield d

    @classmethod
    def _create_loop_orders(cls):
        order = [cls._create_loop(s) for s in Dimension]
        orders = [
            (order[4], order[2], order[3], order[0], order[1]),
            (order[4], order[2], order[3], order[1], order[0]),
            (order[1], order[0], order[4], order[2], order[3]),
        ]
        return orders

    @classmethod
    def _create_loop(cls, dim: Dimension):
        param = cls._param
        if dim == Dimension.Batch:
            size = param.batch
        elif dim == Dimension.InChan:
            size = param.op.in_chan
        elif dim == Dimension.OutChan:
            size = param.op.out_chan
        elif dim == Dimension.Height:
            size = param.height
        elif dim == Dimension.Width:
            size = param.width

        return Loop(dim, size)

    @abstractclassmethod
    def _insert_mem_ops(cls, order: Tuple[Loop]):
        raise NotImplementedError


class ConvDataflowGenerator(DataflowGeneratorFactory):
    @classmethod
    def __insert_weight_load(cls, loop: Loop, checker: dict, visit: dict, outers: list):
        if (
            not checker[DataType.Weight]
            and visit[Dimension.InChan]
            and visit[Dimension.OutChan]
        ):
            param = cls._param
            size = param.op.in_chan * param.op.out_chan
            data = ConvWeightData(size, param)
            loop.mem_ops.append(MemOp(data, MemOpType.LOAD, outers))
            checker[DataType.Weight] = True

    @classmethod
    def __insert_input_load(cls, loop: Loop, checker: dict, visit: dict, outers: list):
        if all(
            [
                not checker[DataType.Input],
                visit[Dimension.InChan],
                visit[Dimension.Batch],
                visit[Dimension.Width],
                visit[Dimension.Height],
            ]
        ):
            param = cls._param
            size = param.batch * param.op.in_chan * param.width * param.height
            data = InputData(size, param)
            loop.mem_ops.append(MemOp(data, MemOpType.LOAD, outers))
            checker[DataType.Input] = True

    @classmethod
    def __insert_output_ops(cls, loop: Loop, checker: dict, visit: dict, outers: list):
        if all(
            [
                not checker[DataType.Output],
                visit[Dimension.OutChan],
                visit[Dimension.Batch],
                visit[Dimension.Width],
                visit[Dimension.Height],
            ]
        ):
            param = cls._param
            size = param.batch * param.op.out_chan * param.width * param.height
            data = OutputData(size, param)
            loop.mem_ops.append(MemOp(data, MemOpType.LOAD, outers))
            loop.mem_ops.append(MemOp(data, MemOpType.STORE, outers))
            checker[DataType.Output] = True

    @classmethod
    def _insert_mem_ops(cls, order: Tuple[Loop]):
        visit = {s: False for s in Dimension}
        checker = {
            DataType.Weight: False,
            DataType.Input: False,
            DataType.Output: False,
        }
        outers = []
        for loop in order:
            loop.mem_ops = []
        for loop in order:
            outers.append(loop)
            visit[loop.dim] = True

            cls.__insert_weight_load(loop, checker, visit, outers)
            cls.__insert_input_load(loop, checker, visit, outers)
            cls.__insert_output_ops(loop, checker, visit, outers)


class PoolDataflowGenerator(DataflowGeneratorFactory):
    pass


if __name__ == "__main__":
    op_param = OPParam()
    param = Param(op=op_param)
    model = GEKKO(remote=False)

    tf = {
        i: model.Var(value=8, lb=8, integer=True, name=i) for i in ["N", "C", "H", "W"]
    }
    tf = TilingFactor(**tf)

    for order in ConvDataflowGenerator.create_generator(param):
        print("==================================")
        loop: Loop
        for loop in order:
            loop.set_tiling_factor(tf)
            print(loop)
