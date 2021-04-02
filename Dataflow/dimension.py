from enum import IntEnum, unique


@unique
class Dimension(IntEnum):
    InChan = 0
    OutChan = 1
    Height = 2
    Width = 3
    Batch = 4
