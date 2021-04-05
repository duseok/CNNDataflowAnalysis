# CNN Dataflow Analysis (CDA)

CNN Dataflow Analysis (CDA) takes convolution layer information and on-chip memory size as inputs. Then, it analyzes the total volume of off-chip data communication and reduces it using loop tiling and loop ordering. The key idea comes from the [SmartShuttle](https://ieeexplore.ieee.org/document/8342033). However, unlike [SmartShuttle](https://ieeexplore.ieee.org/document/8342033), which uses a heuristic algorithm to optimize, CDA searches for the optimal solution using [the GEKKO optimization library](https://gekko.readthedocs.io/en/latest/).

## Table of Contents

- [CNN Dataflow Analysis (CDA)](#cnn-dataflow-analysis-cda)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
    - [Libraries](#libraries)
    - [Quick Install](#quick-install)
  - [Usage](#usage)
    - [Options](#options)
    - [Example](#example)
  - [Results](#results)
    - [VGG16](#vgg16)

## Requirements

### Libraries

- python3
- argparse
- [attrs](https://www.attrs.org/en/stable/)
- [gekko](https://gekko.readthedocs.io/en/latest/)
- [pyyaml](https://pyyaml.org/)

### Quick Install

If you have the 'sudo' privilege,

```zsh
sudo -H pip3 install -r requirements.txt
```

If not,

```zsh
pip3 install --user -r requirements.txt
```

## Usage

### Options

1. `-n` or `--network`: Network configuration file

    This file contains network information, including name, batch size, and layers. Currently, CDA only supports convolution operations. This file is written in YAML.

2. `-s` or `--size` (default=108): On-chip memory size (KB)

3. `-d` or `--dir` (default=results/_{network name}_/): Search result folder

    The search results of each layer are saved in this folder. This file is also written in YAML.

    The _network name_ in the default path comes from the network configuration file.

### Example

```zsh
PYTHONPATH=`pwd` python3 main.py -n input/vgg16.yml -d results/vgg16_108k -s 108
```

## Results

### VGG16

The table below shows the results of optimizing the VGG16 network. All parameters come from [SmartShuttle paper](https://ieeexplore.ieee.org/document/8342033), including compression ratio. Each row, except the first row, shows each layer's optimization results, and each column indicates the results according to the on-chip memory size change.
All off-chip access volume units are MiB.

|            | 108 KiB | 256 KiB | 512 KiB | 768 KiB | 1 MiB |
|:----------:|--------:|--------:|--------:|--------:|------:|
| **Conv1**  |    8.41 |    8.41 |    8.41 |    8.41 |  8.41 |
| **Conv2**  |   14.10 |   13.68 |   13.50 |   13.43 | 13.39 |
| **Conv3**  |    6.50 |    6.15 |    6.06 |    6.03 |  6.01 |
| **Conv4**  |    9.98 |    7.80 |    7.50 |    7.42 |  7.38 |
| **Conv5**  |    5.15 |    3.72 |    3.04 |    2.99 |  2.97 |
| **Conv6**  |    6.18 |    4.91 |    4.06 |    3.53 |  3.44 |
| **Conv7**  |    7.23 |    5.71 |    4.53 |    3.49 |  3.40 |
| **Conv8**  |    3.16 |    2.33 |    2.25 |    1.93 |  1.89 |
| **Conv9**  |    4.54 |    3.67 |    3.23 |    3.23 |  2.99 |
| **Conv10** |    4.48 |    3.88 |    3.57 |    3.08 |  2.72 |
| **Conv11** |    2.72 |    2.32 |    2.20 |    1.47 |  1.36 |
| **Conv12** |    2.28 |    1.89 |    1.86 |    1.21 |  1.14 |
| **Conv13** |    2.75 |    2.02 |    2.00 |    1.36 |  1.28 |
| **Total**  |   77.49 |   66.48 |   62.22 |   57.59 | 56.39 |
