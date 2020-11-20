# GPA
**G**PU **P**erformance **A**dvisor

[![DOI](https://zenodo.org/badge/279470056.svg)](https://zenodo.org/badge/latestdoi/279470056)

GPA is a performance advisor for NVIDIA GPUs that suggests potential code optimization opportunities at a hierarchy of levels, including individual lines, loops, and functions. GPA uses data flow analysis to approximately attribute measured instruction stalls to their root causes and uses information about a program's structure and the GPU to match inefficiency patterns with suggestions for optimization. GPA estimates each optimization's speedup based on a PC sampling-based performance model.

## Quick Start

```bash
git clone --recursive git@github.com:Jokeren/GPA.git && cd GPA
./bin/install.sh
./bin/bench.sh rodinia/bfs
```

## Documentation

- [Installation Guide](https://github.com/Jokeren/GPA/blob/master/INSTALL.md)
- [User's Guide](https://github.com/Jokeren/GPA/tree/master/docs/MANUAL.md)
- [Developer's Guide]
