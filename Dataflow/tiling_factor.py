import attr

from gekko.gk_variable import GKVariable


@attr.s(slots=True, init=True)
class TilingFactor:
    N = attr.ib(type=GKVariable)
    C = attr.ib(type=GKVariable)
    H = attr.ib(type=GKVariable)
    W = attr.ib(type=GKVariable)
