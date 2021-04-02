import attr

from Dataflow.data import DataType


@attr.s(slots=True, init=True)
class HWParam:
    onchip_mem_size = attr.ib(default=108 * 1024)


@attr.s(slots=True, init=True)
class OPParam:
    out_chan = attr.ib(default=128)
    in_chan = attr.ib(default=64)
    stride = attr.ib(default=1)
    kernel = attr.ib(default=3)


@attr.s(slots=True, init=True)
class Param:
    batch = attr.ib(default=3)
    height = attr.ib(default=112)
    width = attr.ib(default=112)
    compression_ratio = attr.ib(
        default={DataType.Input: 1, DataType.Output: 1, DataType.Weight: 1}
    )
    op = attr.ib(default=OPParam())
    hw = attr.ib(default=HWParam())
