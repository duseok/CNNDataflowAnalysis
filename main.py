from __future__ import annotations

import argparse
from functools import reduce
from sys import maxsize
from typing import TYPE_CHECKING

import yaml
from Dataflow.data import DataType
from Dataflow.dataflow import ConvDataflowGenerator
from Dataflow.tiling_factor import TilingFactor
from gekko import GEKKO
from Optimizer.off_chip_access import OffChipAccess
from Optimizer.optimizer import Optimizer
from param import HWParam, OPParam, Param

if TYPE_CHECKING:
    from typing import Dict, Tuple


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--network", type=str, help="network file path", required=True
    )
    parser.add_argument(
        "-d", "--dir", type=str, help="results folder path", default=None
    )
    parser.add_argument(
        "-s",
        "--mem_size",
        type=int,
        help="on-chip memory size (unit: KiB)",
        default=108,
    )
    return parser.parse_args()


def decode_net(file: str):
    with open(file) as f:
        net_dict = yaml.load(f, Loader=yaml.FullLoader)
    return net_dict


def make_layer_info(batch: int, layer: Dict, size: int) -> Param:
    comp_ratio = {}
    for k, v in layer["compression"].items():
        if k == "input":
            comp_ratio[DataType.Input] = v
        elif k == "output":
            comp_ratio[DataType.Output] = v
        elif k == "weight":
            comp_ratio[DataType.Weight] = v
    op_param = OPParam(
        in_chan=layer["in_chan"],
        out_chan=layer["out_chan"],
        kernel=layer["kernel"],
        stride=layer["stride"],
    )
    hw_param = HWParam(size * 1024)
    param = Param(
        batch=batch,
        height=layer["height"],
        width=layer["width"],
        op=op_param,
        hw=hw_param,
        compression_ratio=comp_ratio,
    )
    return param


def main(net_dict: Dict, args):
    if not args.dir:
        args.dir = f"results/{net_dict['name']}"
    size = 0
    for layer in net_dict["layers"]:
        param = make_layer_info(net_dict["batch"], layer, args.mem_size)
        print(f"Layer: {layer['name']:10s}", end="\t")
        _, min_val = results = search(param)
        size += min_val
        print_solutions(results, args, layer["name"])
    print(f"Total Size (MiB): {size/1024/1024:.2f}")


def search(param: Param):
    min_val = maxsize
    solutions = []
    for order in ConvDataflowGenerator.create_generator(param):
        model, tf = initialize()
        opt = Optimizer(model)
        obj = OffChipAccess(opt, param, order, tf)
        obj.set_search_spase()
        tf, size = opt.solve()
        if size != maxsize and size <= min_val:
            solutions = solutions if size == min_val else []
            solutions.append((order, tf))
            min_val = size
    print(f"# of Solutions: {len(solutions)}\tSize (MiB): {min_val/1024/1024:.2f}")
    return solutions, min_val


def print_solutions(results: Tuple, args, name: str):
    from pathlib import Path

    solutions, val = results
    dir = args.dir
    Path(dir).mkdir(parents=True, exist_ok=True)
    with open(f"{dir}/{name}.yaml", "w") as f:
        ret = {"Result (KiB)": float(f"{val / 1024:.2f}"), "Solutions": []}
        for order, tf in solutions:
            order_str = reduce(lambda x, y: x + y.dim.name[0], ["", *order])
            tf_str = {k: int(v) for k, v in tf.items() if k[-1] in "nchw"}
            ret["Solutions"].append({"Order": order_str, "Tiling Factor": tf_str})
        yaml.dump(ret, f)


def initialize() -> Tuple[GEKKO, TilingFactor]:
    model = GEKKO(remote=False)
    # XXX minimum values are fixed
    tf = {i: model.Var(1, 1, integer=True, name=i) for i in ["N", "C", "H", "W"]}
    tf = TilingFactor(**tf)
    return model, tf


if __name__ == "__main__":
    args = parse()
    net_dict = decode_net(args.network)
    main(net_dict, args)
