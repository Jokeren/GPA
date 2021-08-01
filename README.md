# GPA
**G**PU **P**erformance **A**dvisor

[![DOI](https://zenodo.org/badge/279470056.svg)](https://zenodo.org/badge/latestdoi/279470056)

GPA is a performance advisor for NVIDIA GPUs that suggests potential code optimization opportunities at a hierarchy of levels, including individual lines, loops, and functions. GPA uses data flow analysis to approximately attribute measured instruction stalls to their root causes and uses information about a program's structure and the GPU to match inefficiency patterns with suggestions for optimization. GPA estimates each optimization's speedup based on a PC sampling-based performance model.

## Quick Start

```bash
git clone --recursive https://github.com/Jokeren/GPA.git && cd GPA
./bin/install.sh
./bin/bench.sh rodinia/bfs
```

## Documentation

- [Installation Guide](https://github.com/Jokeren/GPA/blob/master/INSTALL.md)
- [User's Guide](https://github.com/Jokeren/GPA/tree/master/docs/MANUAL.md)
- [Developer's Guide]

## Papers

- K. Zhou, X. Meng, R. Sai, D. Grubisic and J. Mellor-Crummey, "An Automated Tool for Analysis and Tuning of GPU-accelerated Code in HPC Applications." *IEEE Transactions on Parallel and Distributed Systems* (TPDS) (2021).
- K. Zhou, X. Meng, R. Sai and J. Mellor-Crummey, "GPA: A GPU Performance Advisor Based on Instruction Sampling," *2021 IEEE/ACM International Symposium on Code Generation and Optimization* (CGO), Seoul, Korea (South), 2021, pp. 115-125, doi: 10.1109/CGO51591.2021.9370339.
